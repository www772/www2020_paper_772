import torch
import torch.nn as nn
import random
import string
import numpy as np

class dualpc(nn.Module): 
    def __init__(self, conf):
        super(dualpc, self).__init__()
        self.conf = conf
        self.nb_digits = 5
        self.rating_marginal_probability = - 0.9189
        #self.review_marginal_probability = - 1.9407
        self.review_marginal_probability = - 1.9000
        self.initializeNodes()
        self.defineMap()
        
    def setDevice(self, device):
        self.device = device

        self.initializeWordEmbedding()
        #self.initalizeUserAndItemEmbedding()

    def tensorToScalar(self, tensor):
        return tensor.cpu().detach().numpy()
    
    # return rating_list[0] * 5
    def one_hot(self, rating_list):
        rating_list = torch.reshape(rating_list, (-1, 1)) - 1.0
        # rating_list size: batch_size * 1
        batch_size = rating_list.size()[0]
        y_onehot = torch.zeros(batch_size, self.nb_digits).to(self.device)
        
        # y_onehot is the matrix after one-hot encoding
        y_onehot.scatter_(1, rating_list, 1)
        return y_onehot
        
    def initializeNodes(self):
        self.initalizeR2ENodes()
        self.initalizeE2RNodes()

    def initalizeR2ENodes(self):
        torch.manual_seed(0)
        self.r2e_item_embedding = nn.Embedding(self.conf.r2e_num_items, self.conf.r2e_mf_dimension)
        torch.manual_seed(0)
        self.r2e_user_embedding = nn.Embedding(self.conf.r2e_num_users, self.conf.r2e_mf_dimension)

        torch.manual_seed(0)
        self.r2e_rating_to_initial_state_mapping_layer = \
            nn.Linear(self.nb_digits, self.conf.r2e_hidden_dimension)
        torch.manual_seed(0)
        self.r2e_user_to_initial_state_mapping_layer = \
            nn.Linear(self.conf.r2e_mf_dimension, self.conf.r2e_hidden_dimension)
        torch.manual_seed(0)
        self.r2e_item_to_initial_state_mapping_layer = \
            nn.Linear(self.conf.r2e_mf_dimension, self.conf.r2e_hidden_dimension)

        torch.manual_seed(0)
        self.r2e_word_embedding = nn.Embedding(self.conf.r2e_num_words + 3, self.conf.r2e_word_dimension)
        torch.manual_seed(0)
        self.r2e_lstm = \
            nn.GRU(self.conf.r2e_word_dimension, self.conf.r2e_hidden_dimension, num_layers=1, bidirectional=False)

        torch.manual_seed(0)
        self.r2e_lstm_output_mapping_layer = \
            nn.Linear(self.conf.r2e_hidden_dimension, self.conf.r2e_num_words + 3)
    
    def initializeWordEmbedding(self):
        self.r2e_basic_word_embedding_weight = \
            torch.FloatTensor(np.load(self.conf.r2e_word_embedding)).to(self.device)
        # initialize three word embedding for SOS, EOS, PAD respectively
        self.r2e_extra_word_embedding_weight = \
            torch.FloatTensor(np.random.rand(3, self.conf.r2e_word_dimension)).to(self.device)

        self.r2e_word_embedding_weight = \
            torch.cat((self.r2e_basic_word_embedding_weight, self.r2e_extra_word_embedding_weight), 0)
        self.r2e_word_embedding.weight = nn.Parameter(self.r2e_word_embedding_weight)

    def initalizeUserAndItemEmbedding(self):
        r2e_model_parameters = torch.load(self.conf.r2e_assist_pre_model)
        
        r2e_item_embedding_weight = r2e_model_parameters['gmf_item_embedding.weight'].to(self.device)
        self.r2e_item_embedding.weight = nn.Parameter(r2e_item_embedding_weight)
        self.r2e_item_embedding.weight.requires_grad = False

        r2e_user_embedding_weight = r2e_model_parameters['gmf_user_embedding.weight'].to(self.device)
        self.r2e_user_embedding.weight = nn.Parameter(r2e_user_embedding_weight)
        self.r2e_user_embedding.weight.requires_grad = False

    def initalizeE2RNodes(self):
        torch.manual_seed(0)
        self.e2r_gmf_user_embedding = nn.Embedding(self.conf.num_users, self.conf.e2r_mf_dimension)
        self.e2r_gmf_user_embedding.weight = nn.Parameter(0.1 * self.e2r_gmf_user_embedding.weight)
        torch.manual_seed(0)
        self.e2r_gmf_item_embedding = nn.Embedding(self.conf.num_items, self.conf.e2r_mf_dimension)
        self.e2r_gmf_item_embedding.weight = nn.Parameter(0.1 * self.e2r_gmf_item_embedding.weight)

        self.e2r_relu = nn.ReLU()

        # review representation mapping layer
        torch.manual_seed(0)
        self.e2r_review_mapping_layer = nn.Linear(self.conf.e2r_text_char_dimension, self.conf.e2r_mf_dimension)

        torch.manual_seed(0)
        self.e2r_gmf_bn = nn.BatchNorm1d(self.conf.e2r_mf_dimension)
        torch.manual_seed(0)
        self.e2r_mlp_bn = nn.BatchNorm1d(self.conf.e2r_mf_dimension)
        torch.manual_seed(0)
        self.e2r_review_bn = nn.BatchNorm1d(self.conf.e2r_mf_dimension)

    def initOptimizer(self):
        self.initR2EOptimizer()
        self.initE2ROptimizer()

    def initR2EOptimizer(self):
        self.r2e_criterion = nn.CrossEntropyLoss()
        self.r2e_postprior_calculator = nn.CrossEntropyLoss(reduction='none')

        r2e_trainable_parameters = []
        for (key, value) in self._modules.items():
            if 'r2e' in key:
                r2e_trainable_parameters += list(value.parameters())

        self.r2e_optimizer = torch.optim.Adam(r2e_trainable_parameters)

    def initE2ROptimizer(self):
        e2r_trainable_parameters = []
        for (key, value) in self._modules.items():
            if 'e2r' in key:
                e2r_trainable_parameters += list(value.parameters())

        self.e2r_opt_criterion = nn.MSELoss(reduction='sum')
        self.e2r_optimizer = torch.optim.Adam(e2r_trainable_parameters, weight_decay=0.1)

    def train(self, feed_dict):
        self.user_input = torch.LongTensor(feed_dict['USER_INPUT']).to(self.device)
        self.item_input = torch.LongTensor(feed_dict['ITEM_INPUT']).to(self.device)
        self.label_input = torch.LongTensor(feed_dict['RATING_INPUT']).to(self.device)

        self.e2r_label_input = torch.FloatTensor(feed_dict['RATING_INPUT']).to(self.device)

        self.e2r_review_representation_input = \
            torch.FloatTensor(feed_dict['REVIEW_REPRESENTATION_INPUT']).to(self.device)

        # following inputs for the r2e model
        self.word_input = torch.LongTensor(feed_dict['WORD_INPUT']).to(self.device)
        self.target_input = torch.LongTensor(feed_dict['TARGET_INPUT']).to(self.device)

        # start to compute e2r & r2e loss
        self.e2r()
        self.r2e()
        self.computeDualityLoss()
        self.computeE2RLoss()
        self.computeR2ELoss()
        self.defineTrainOutMap()

    def e2r(self):
        # compute gmf part
        e2r_gmf_user_vector = self.e2r_gmf_user_embedding(self.user_input) #batch_size * dimension
        e2r_gmf_item_vector = self.e2r_gmf_item_embedding(self.item_input) #batch_size * dimension
        e2r_mul_gmf_vector = e2r_gmf_user_vector * e2r_gmf_item_vector #batch_size * dimension
        e2r_mul_gmf_vector = self.e2r_gmf_bn(e2r_mul_gmf_vector)

        '''We will concatenate the information here, the information contains the cf/review'''
        #review_representation_vector = self.review_bn(review_representation_input)
        e2r_review_mapping_vector = self.e2r_relu(self.e2r_review_mapping_layer(self.e2r_review_representation_input))
        e2r_review_mapping_vector = self.e2r_review_bn(e2r_review_mapping_vector)

        e2r_final_concat_vector = \
            torch.cat((e2r_mul_gmf_vector, e2r_review_mapping_vector), 1)

        '''SHOULD I TRY SOME NON-LINEAR FUNCTION HERE?'''
        self.e2r_tmp_prediction = torch.sum(e2r_final_concat_vector, 1) #size: batch * 5
        self.e2r_rating_prediction = self.e2r_tmp_prediction

        self.e2r_rating_prediction = torch.clamp(self.e2r_rating_prediction, min=1.0, max=5.0)

        self.e2r_opt_loss = self.e2r_opt_criterion(self.e2r_rating_prediction, self.e2r_label_input)

        # compute the post prior probability
        # e2r p(\hat{r}_ai|C_ai, a, i)
        self.e2r_postprior = \
            torch.pow((self.e2r_rating_prediction - self.e2r_label_input), 2).mul(-0.5).add(-0.9189)

    def r2e(self):
        r2e_rating_vector = self.one_hot(self.label_input)
        r2e_rating_vector = self.r2e_rating_to_initial_state_mapping_layer(r2e_rating_vector)
        r2e_user_vector = self.r2e_user_embedding(self.user_input)
        r2e_user_vector = self.r2e_user_to_initial_state_mapping_layer(r2e_user_vector)
        r2e_item_vector = self.r2e_item_embedding(self.item_input)
        r2e_item_vector = self.r2e_item_to_initial_state_mapping_layer(r2e_item_vector)

        r2e_cat_vector = r2e_rating_vector + r2e_user_vector + r2e_item_vector

        r2e_initial_state = torch.reshape(r2e_cat_vector, (1, -1, self.conf.r2e_hidden_dimension))

        r2e_word_vector = self.r2e_word_embedding(self.word_input) #size: (time*batch) * self.conf.text_word_dimension
        #word_vector = self.bn_1(word_vector)
        r2e_reshape_word_vector = torch.reshape(\
            r2e_word_vector, (self.conf.sequence_length, -1, self.conf.r2e_word_dimension)) # time * batch * word_dimension
        r2e_outputs, _ = self.r2e_lstm(r2e_reshape_word_vector, r2e_initial_state) # time * batch * hidden_dimension
        # convert the outputs(dimension: hidden_dimension) to dimension: num_words + 3
        r2e_outputs_2 = self.r2e_lstm_output_mapping_layer(r2e_outputs) # time * batch * (num_words + 3)
        #outputs_2 = self.bn_2(outputs_2)
        # compute the similarity of the outputs and corresponding vocab
        r2e_reshape_outputs = torch.reshape(r2e_outputs_2, (-1, self.conf.r2e_num_words + 3))
        #reshape_outputs = self.bn_3(reshape_outputs)
        r2e_word_probit = r2e_reshape_outputs

        self.r2e_opt_loss = self.r2e_criterion(r2e_word_probit, self.target_input)

        # r2e p(\hat{C}_ai|r_ai)
        # calculate the postproior of the r2e model
        r2e_words_postprior = self.r2e_postprior_calculator(r2e_word_probit, self.target_input)
        # following is used to compute the sum log probability of each sentence
        r2e_words_postprior = torch.reshape(r2e_words_postprior, (self.conf.sequence_length, -1)) #time * batch
        # 1*batch, nll_of_all_words[idx] denotes the negative likelihood loss of the sentence idx, -\sum_{t=0}^T log p(C^t_ai)
        self.r2e_postprior = torch.reshape(torch.mean(r2e_words_postprior, 0), [-1, 1]) # batch * 1

    def computeDualityLoss(self):
        # the initial set of rating marginal probability and review marginal probability is negative value
        # should be the correct log(rating) and log(review)
        # ( log(p(r_ai)) + log(p(C_ai|r_ai)) ) - ( log(p(C_ai)) + log(p(r_ai|C_ai)) ) = 
        # ( log(p(r_ai)) - (-log(p(C_ai|r_ai))) ) - ( (log(p(C_ai))) + (-log(p(r_ai|C_ai))) )
        self.basic_term = (self.rating_marginal_probability + (-self.r2e_postprior)) - \
            (self.review_marginal_probability + self.e2r_postprior)
        duality_loss = torch.pow(self.basic_term, 2)
        self.duality_loss = torch.mean(duality_loss)

    def computeE2RLoss(self):
        self.final_e2r_loss = self.e2r_opt_loss + 0.2 * self.duality_loss
    
    def computeR2ELoss(self):
        self.final_r2e_loss = self.r2e_opt_loss + 0.2 * self.duality_loss

    def optimizeE2R(self):
        self.zero_grad()
        self.final_e2r_loss.backward()
        self.e2r_optimizer.step()
    
    def optimizeR2E(self):
        self.zero_grad()
        self.final_r2e_loss.backward()
        self.r2e_optimizer.step()

    def defineMap(self):
        map_dict = {}

        map_dict['input'] = {
            'WORD_INPUT': 'WORD_LIST', 
            'TARGET_INPUT': 'TARGET_LIST',
            'USER_INPUT': 'USER_LIST',
            'ITEM_INPUT': 'ITEM_LIST',
            'RATING_INPUT': 'RATING_LIST',
            'REVIEW_REPRESENTATION_INPUT': 'REVIEW_REPRESENTATION_LIST'
        }

        self.map_dict = map_dict
    
    def defineTrainOutMap(self):
        self.map_dict['out'] = {
            'e2r_opt_loss': self.tensorToScalar(self.e2r_opt_loss),
            'e2r_output_loss': self.tensorToScalar(self.e2r_opt_loss),
            'r2e_opt_loss': self.tensorToScalar(self.r2e_opt_loss),
            'duality_loss': self.tensorToScalar(self.duality_loss),
            'e2r_prediction': self.tensorToScalar(self.e2r_rating_prediction),
            'e2r_tmp_prediction': self.tensorToScalar(self.e2r_tmp_prediction)
        }