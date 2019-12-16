import numpy as np

class DataModule():
    def __init__(self, conf, filename,
                 review_representation_path='',
                 interaction_representation_path = '',
                 rating_marginal_distribution = '',
                 review_marginal_distribution='', 
                 user_item_review_representation_path = ''):
        self.index, self.terminal_flag, self.data_dict = 0, 1, {}
        self.conf, self.filename = conf, filename
        self.user_item_review_representation_path = user_item_review_representation_path
        self.review_representation_path = review_representation_path
        self.rating_marginal_distribution = rating_marginal_distribution
        self.review_marginal_distribution = review_marginal_distribution

########################################### Task 16 Data Preparation ######################################
    def task16InitializeData(self, model):
        # define some necessary flags
        self.SOS = self.conf.r2e_num_words
        self.EOS = self.conf.r2e_num_words + 1
        self.PAD = self.conf.r2e_num_words + 2

        self.model = model
        self.data_read_pointer = 0
        self.data_write_pointer = 0
        self.terminal_flag = 1

        self.readDictionary()
        self.task15ReadWords()
        self.readReviewRepresentation()
        self.task16CreateTrainBatches()

    # get the train/val/test representation paths first
    def readReviewRepresentation(self):
        self.review_representation = np.load(self.review_representation_path).item()

    # convert the sequence of words into ids
    def tokenizeWord(self, review, target_flag):
        tokenize_list = []
        if target_flag == 'input':
            tokenize_list = [self.SOS]
            tokenize_list.extend(review)
            tokenize_list.append(self.EOS)
        elif target_flag == 'target':
            tokenize_list = []
            tokenize_list.extend(review)
            tokenize_list.append(self.EOS)

        #print(tokenize_list)
        #print(review)
        #pad = tokenize_list[-1]
        pad = self.PAD
        tokenize_list.extend([pad] * (self.conf.sequence_length - len(tokenize_list)))
        #print(tokenize_list)

        return tokenize_list
        
    def task15ReadWords(self):
        origin_data_dict, tokenize_data_dict = {}, {}
        with open(self.filename) as f:
            for line in f:
                line = eval(line)
                user, item, rating, review, idx = \
                    line['user_id'], line['item_id'], line['rating'], line['review'], line['idx']
                #rating = (rating - 1.0)/4.0
                #get the training review with maximum length self.conf.sequence_length
                origin_data_dict[idx] = [user, item, rating, review]
                tokenize_review = self.tokenizeWord(review, target_flag='input')
                tokenize_data_dict[idx] = [user, item, rating, tokenize_review]
        self.origin_data_dict = origin_data_dict
        self.tokenize_data_dict = tokenize_data_dict

    # Construct key: char-> value: idx, key: idx-> value: char vocab
    def readDictionary(self):
        self.vocab = np.load(self.conf.vocab).item()
        self.decoder_vocab = np.load(self.conf.decoder_vocab).item()

        # update self.vocab and self.decoder_vocab
        self.vocab['SOS'] = self.SOS
        self.vocab['EOS'] = self.EOS
        self.vocab['PAD'] = self.PAD

        self.decoder_vocab[self.SOS] = 'SOS'
        self.decoder_vocab[self.EOS] = 'EOS'
        self.decoder_vocab[self.PAD] = 'PAD'

    # this function can be regarded as the boost version of creating training batches
    def task16CreateTrainBatches(self):
        user_dict, item_dict, rating_dict, input_dict, target_dict = {}, {}, {}, {}, {}
        for (idx, record) in self.origin_data_dict.items():
            [user, item, rating, review] = record
            input_dict[idx] = self.tokenizeWord(review, target_flag='input')
            target_dict[idx] = self.tokenizeWord(review, target_flag='target')

            user_dict[idx] = user
            item_dict[idx] = item
            rating_dict[idx] = rating
        self.input_dict = input_dict
        self.target_dict = target_dict
        self.rating_dict = rating_dict
        self.user_dict = user_dict
        self.item_dict = item_dict

        # prepare for all the training batches
        self.data_to_be_input = {} # Which is used to store all the training batches
        
        index, total_batch_size = 0, 0
        total_batch_idx_list = list(self.origin_data_dict.keys())
        total_length = len(total_batch_idx_list)
        while (index + self.conf.training_batch_size < total_length):
            batch_idx_list = total_batch_idx_list[index:index + self.conf.training_batch_size]
            total_batch_size += self.task16ConstructBatchBundle(batch_idx_list)
            index += self.conf.training_batch_size
            self.data_write_pointer += 1
        batch_idx_list = total_batch_idx_list[index:]
        total_batch_size += self.task16ConstructBatchBundle(batch_idx_list)
        self.data_write_pointer += 1
        print('load %s, total_batch_size: %s' % (self.filename, total_batch_size))
        
    # This function is used to construct the training batch bundle
    def task16ConstructBatchBundle(self, batch_idx_list):
        word_list, target_list, rating_list, \
            user_list, item_list = [], [], [], [], []
        review_representation_list = []
        for idx in batch_idx_list:
            word_list.append(self.input_dict[idx])
            target_list.append(self.target_dict[idx])
            
            user_list.append(self.user_dict[idx])
            item_list.append(self.item_dict[idx])
            rating_list.append(self.rating_dict[idx])
            review_representation_list.append(self.review_representation[idx])

        # in order to transpose the origin matrix into time-based
        # !Important, batch-based is not working
        word_list = np.transpose(word_list)
        word_list = np.reshape(word_list, (-1)) # 1 * (T*B)

        target_list = np.transpose(target_list)
        target_list = np.reshape(target_list, (-1)) # 1 * (T*B)
        
        rating_list = np.reshape(rating_list, (-1)) # used as the index
        user_list = np.array(user_list)
        item_list = np.array(item_list)

        review_representation_list = np.array(review_representation_list)

        current_batch_size = len(batch_idx_list)
        self.data_to_be_input[self.data_write_pointer] = \
            [user_list, item_list, rating_list, word_list, target_list, \
                review_representation_list, batch_idx_list]
        return current_batch_size

    # this function is bundled with the boost version of task16CreateTrainBatches
    def task16GetBatch(self):
        [self.user_list, self.item_list, self.rating_list, self.word_list, self.target_list, \
            self.review_representation_list, self.batch_idx_list] =\
            self.data_to_be_input[self.data_read_pointer]
        self.task16CreateMap()
        self.data_read_pointer += 1 # Important, update data read pointer

        if self.data_read_pointer == self.data_write_pointer:
            self.terminal_flag = 0
            self.data_read_pointer = 0

    def task16CreateMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['RATING_LIST'] = self.rating_list

        self.data_dict['WORD_LIST'] = self.word_list
        self.data_dict['TARGET_LIST'] = self.target_list

        self.data_dict['REVIEW_REPRESENTATION_LIST'] = self.review_representation_list