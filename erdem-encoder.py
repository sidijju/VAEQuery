import torch
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_FEATURES = 4
RATIONALITY = 1.
BATCH_SIZE = 64
SAMPLE_BELIEF = True
SEQUENCE_LENGTH = 20

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # fc layer before passing into GRU units
        # TODO: add option to make multiple layers
        self.hidden_dim = 64

        self.fc_embedding = nn.Linear(NUM_FEATURES, 16)
        
        self.fc_correct = nn.Linear(16, 64)
        self.fc_rest = nn.Linear(16, 64)

        # RNN functionality
        self.gru = nn.GRU(input_size=128,
                          hidden_size=self.hidden_dim,
                          num_layers=1)

        # fc layers after the RNN
        fc_layers = []
        curr = self.hidden_dim
        for l in [32]:
            fc_layers.append(nn.Linear(curr, l))
            curr = l
        self.fc_layers = fc_layers

        self.fc_mu = nn.Linear(curr, NUM_FEATURES)
        self.fc_logvar = nn.Linear(curr, NUM_FEATURES)

    def forward(self, query, hidden):
        # embed all query inputs
        embeddings = self.fc_embedding(query)

        # separate first query from the rest
        correct_embedding = embeddings[:, :, 0, :]
        rest_embeddings = embeddings[:, :, 1, :]
        #correct_embedding = torch.flatten(correct_embedding, start_dim=-2)
        #rest_embeddings = torch.flatten(rest_embeddings, start_dim=-2)

        # fc layer
        correct = self.fc_correct(correct_embedding)
        correct = F.leaky_relu(correct)
        rest = self.fc_rest(rest_embeddings)
        rest = F.leaky_relu(rest)
        output = torch.cat((correct, rest), dim=-1)

        # run through gru
        output, hidden = self.gru(output, hidden)

        # run through fc layers
        for l in self.fc_layers:
            output = l(output)
            output = F.leaky_relu(output)

        # output belief distribution
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)

        return mu, logvar, hidden

    def init_hidden(self, batchsize):
        hidden = torch.zeros((1, batchsize, self.hidden_dim))
        return hidden
        
def generate_reward(num_rewards=1):
    vecs = torch.randn(num_rewards, NUM_FEATURES)
    vecs = vecs / torch.linalg.norm(vecs, dim=-1)
    return vecs
        
def synthesize_queries(num_queries):
    trajA = torch.randn(num_queries, NUM_FEATURES)
    trajB = torch.randn(num_queries, NUM_FEATURES)
    return trajA, trajB
    
def respond_queries(rewardVec, trajA, trajB, give_dist = False):
    # return 0 for A, 1 for B
    if len(trajA.shape) == 1:
        trajA = trajA.reshape((1, NUM_FEATURES))
        trajB = trajB.reshape((1, NUM_FEATURES))

    assert (len(rewardVec.shape) == 1 and rewardVec.shape[0] == NUM_FEATURES) or (rewardVec.shape[0] == 1 and rewardVec.shape[1] == NUM_FEATURES)
        
    num_queries = len(trajA)
    rewA = (rewardVec @ trajA.T).squeeze()
    rewB = (rewardVec @ trajB.T).squeeze()

    if not give_dist:
        return (rewA < rewB).long() # oracle user here for simplicity
    else:
        return torch.cat((rewA.unsqueeze(1), rewB.unsqueeze(1)), dim=1) * RATIONALITY
 

def decoder_loss(rewardVec, trajA, trajB, mu, logvar):
    loss1 = torch.square(torch.linalg.norm(mu) - 1)
    
    human_answers = respond_queries(rewardVec, trajA, trajB, give_dist = False)
    if SAMPLE_BELIEF:
        sample = (torch.randn(NUM_FEATURES) * torch.sqrt(torch.exp(logvar))) + mu
        predicted_logits = respond_queries(sample, trajA, trajB, give_dist = True)
    else:
        predicted_logits = respond_queries(mu, trajA, trajB, give_dist = True)
    
    loss_func = nn.CrossEntropyLoss()
    loss2 = loss_func(predicted_logits, human_answers)
    
    return loss1 + loss2

def alignment(true_vec, mu, logvar, m=100):
    samples = (torch.randn(m,NUM_FEATURES) * torch.sqrt(torch.exp(logvar))) + mu
    samples = samples / torch.linalg.norm(samples,dim=-1).unsqueeze(1)
    true_vec_T = true_vec.permute(*torch.arange(true_vec.ndim - 1, -1, -1))
    return abs(torch.mean(samples @ true_vec_T))

if __name__ == "__main__":
    encoder = Encoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    hidden = encoder.init_hidden(BATCH_SIZE)
    counter = 0
    losses = []
    while counter < 2000:
        
        true_reward_vec = generate_reward()
        trajA, trajB = synthesize_queries(SEQUENCE_LENGTH * BATCH_SIZE)
        human_answers = respond_queries(true_reward_vec, trajA, trajB, give_dist = False)
        
        for i in range(SEQUENCE_LENGTH * BATCH_SIZE):
            if human_answers[i] > 0.5:
                trajA[i], trajB[i] = trajB[i], trajA[i]
        
        trajA_matrix = trajA.reshape((BATCH_SIZE, 1, SEQUENCE_LENGTH, NUM_FEATURES))
        trajB_matrix = trajB.reshape((BATCH_SIZE, 1, SEQUENCE_LENGTH, NUM_FEATURES))
        
        mu_list = torch.zeros((SEQUENCE_LENGTH * BATCH_SIZE, NUM_FEATURES))
        logvar_list = torch.zeros((SEQUENCE_LENGTH * BATCH_SIZE, NUM_FEATURES))
        seq_list = torch.zeros((SEQUENCE_LENGTH * BATCH_SIZE,))
        true_reward_list = torch.zeros((SEQUENCE_LENGTH * BATCH_SIZE, NUM_FEATURES))
        idx = 0
        
        for query_no in range(SEQUENCE_LENGTH):
            query = torch.cat((trajA_matrix[:,0,query_no,:].unsqueeze(1), trajB_matrix[:,0,query_no,:].unsqueeze(1)), dim=1)
        
            mu, logvar, hidden = encoder(query.unsqueeze(0), hidden)
            mu_list[idx:idx+BATCH_SIZE] = mu
            logvar_list[idx:idx+BATCH_SIZE] = logvar
            seq_list[idx:idx+BATCH_SIZE] = query_no
            true_reward_list[idx:idx+BATCH_SIZE] = true_reward_vec.expand(BATCH_SIZE,NUM_FEATURES)
            idx += BATCH_SIZE
        
        hidden = encoder.init_hidden(BATCH_SIZE)
        
        
        loss = 0
        mid_alignment = 0
        final_alignment = 0
        trajA, trajB = synthesize_queries(64)
        for i in range(SEQUENCE_LENGTH * BATCH_SIZE):
            loss += (seq_list[i] + 1) * decoder_loss(true_reward_list[i], trajA, trajB, mu_list[i], logvar_list[i])
            if seq_list[i] == SEQUENCE_LENGTH//2:
                mid_alignment += alignment(true_reward_list[i], mu_list[i], logvar_list[i])
            elif seq_list[i] == SEQUENCE_LENGTH-1:
                final_alignment += alignment(true_reward_list[i], mu_list[i], logvar_list[i])

        loss /= BATCH_SIZE
        mid_alignment /= BATCH_SIZE
        final_alignment /= BATCH_SIZE
                
        
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        counter += 1
        if counter % 10 == 9:
            print('[Epoch {:5d}]: Loss = {:10.3f}, Mid-Alignment = {:10.3f}, Final-Alignment = {:10.3f}'.format(counter+1, loss.detach().numpy(), mid_alignment.detach().numpy(), final_alignment.detach().numpy()))
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("CE Error")
    plt.title("Query Distribution vs. Answer Error")
    plt.savefig("erdem-train-loss")
    plt.close()

        
        
        
        
        
        
        
        
        