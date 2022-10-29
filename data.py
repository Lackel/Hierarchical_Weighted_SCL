import random
import os
import csv
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


class Data:
    def __init__(self, args):
        self.args = args
        self.set_seed()
        MAX_SEQ_LEN = {'clinc':30, 'wos':200, 'hwu64':10}
        self.max_seq_length = MAX_SEQ_LEN[args.dataset]
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        
        self.label_list_coarse, self.label_list_fine = self.get_label_list() # Coarse and fine label lists.
        self.n_fine = len(self.label_list_fine)
        self.n_coarse = len(self.label_list_coarse)

        self.train_examples = self.get_examples('train')
        self.test_examples = self.get_examples('test')
        self.train_dataloader = self.get_data_loader(self.train_examples, 'train')
        self.test_dataloader = self.get_data_loader(self.test_examples, 'test')

        print('Number of coarse labels:', self.n_coarse)
        print('Number of fine labels:', self.n_fine)
        print('Number of train samples', len(self.train_examples))
        print('Number of test  samples', len(self.test_examples))
    
    def set_seed(self):
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_label_list(self):
        """
        get coarse and fine label lists.
        """
        label_list_coarse = []
        data = self.read_tsv(os.path.join(self.data_dir, "train.tsv"))
        # get unique fine label list.
        label_list_fine = np.unique(np.array([i[1] for i in data], dtype=str))
        for i in data:
            if self.args.dataset == 'clinc':
                label_list_coarse.append(self.clinc_fine_to_coarse(i[1]))
            elif self.args.dataset == 'wos':
                label_list_coarse.append(self.wos_fine_to_coarse(i[1]))
            elif self.args.dataset == 'hwu64':
                label_list_coarse.append(self.hwu_fine_to_coarse(i[1]))
        # get unique coarse label list.
        label_list_coarse = np.unique(np.array(label_list_coarse, dtype=str))
        return label_list_coarse, label_list_fine
   
    def get_examples(self, mode):
        """
        Convert data to examples for subsequent processing.
        """
        lines = self.read_tsv(os.path.join(self.data_dir, mode + '.tsv'))
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                continue
            text = line[0] # get text
            label_fine = line[1] # get fine label
            # get coarse label
            if self.args.dataset == 'clinc':
                label_coarse = self.clinc_fine_to_coarse(label_fine)
            elif self.args.dataset == 'wos':
                label_coarse = self.wos_fine_to_coarse(label_fine)
            elif self.args.dataset == 'hwu64':
                label_coarse = self.hwu_fine_to_coarse(label_fine)
            # print(label_fine)
            examples.append(
                InputExample(text=text, label_coarse=label_coarse, label_fine=label_fine))
        return examples

    def get_data_loader(self, examples, mode):
        """
        Build dataloaders with tokenized features.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, do_lower_case=True)
        features = self.convert_examples_to_features(examples, self.max_seq_length, tokenizer, mode)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_coarse_ids = torch.tensor([f.label_id_coarse for f in features], dtype=torch.long)
        label_fine_ids = torch.tensor([f.label_id_fine for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_coarse_ids, label_fine_ids)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.train_batch_size)    
        elif mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.eval_batch_size) 
        
        return dataloader

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer, mode):
        """
        Convert examples to features (input_ids, input_mask, segment_ids, label_coarse_ids, label_fine_ids)for BERT.
        """
        label_map_coarse = {}
        label_map_fine = {}
        for i, label in enumerate(self.label_list_coarse):
            label_map_coarse[label] = i
        for i, label in enumerate(self.label_list_fine):
            label_map_fine[label] = i

        features = []
        for _, example in enumerate(examples):
            tokens_a = tokenizer(example.text, padding='max_length', max_length=max_seq_length, truncation=True)

            input_ids = tokens_a['input_ids']
            input_mask = tokens_a['attention_mask']
            segment_ids = tokens_a['token_type_ids']
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id_coarse = label_map_coarse[example.label_coarse]
            label_id_fine = label_map_fine[example.label_fine]

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id_coarse=label_id_coarse,
                            label_id_fine=label_id_fine))
        return features

    def read_tsv(self, file):
        """
        Reads origin data (text, fine_label) from tsv files.
        """
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            # skip the headline.       
            return lines[1:]
    
    def clinc_fine_to_coarse(self, fine_label):
        """
        Convert fine labels to coarse ones for CLINC dataset.
        """
        label_dict = {'transfer':'banking', 'transactions':'banking', 'balance':'banking', 'freeze_account':'banking', 'pay_bill':'banking', 'bill_balance':'banking', 'bill_due':'banking', 'interest_rate':'banking', 'routing':'banking', 'min_payment':'banking', 'order_checks':'banking', 'pin_change':'banking', 'report_fraud':'banking', 'account_blocked':'banking', 'spending_history':'banking', 
        'credit_score':'credit_cards', 'report_lost_card':'credit_cards', 'credit_limit':'credit_cards', 'rewards_balance':'credit_cards', 'new_card':'credit_cards', 'application_status':'credit_cards', 'card_declined':'credit_cards', 'international_fees':'credit_cards', 'apr':'credit_cards', 'redeem_rewards':'credit_cards', 'credit_limit_change':'credit_cards', 'damaged_card':'credit_cards', 'replacement_card_duration':'credit_cards', 'improve_credit_score':'credit_cards', 'expiration_date':'credit_cards', 
        'recipe':'kitchen', 'restaurant_reviews':'kitchen', 'calories':'kitchen', 'nutrition_info':'kitchen', 'restaurant_suggestion':'kitchen', 'ingredients_list':'kitchen', 'ingredient_substitution':'kitchen', 'cook_time':'kitchen', 'food_last':'kitchen', 'meal_suggestion':'kitchen', 'restaurant_reservation':'kitchen', 'confirm_reservation':'kitchen', 'how_busy':'kitchen', 'cancel_reservation':'kitchen', 'accept_reservations':'kitchen', 
        'shopping_list':'home', 'shopping_list_update':'home', 'next_song':'home', 'play_music':'home', 'update_playlist':'home', 'todo_list':'home', 'todo_list_update':'home', 'calendar':'home', 'calendar_update':'home', 'what_song':'home', 'order':'home', 'order_status':'home', 'reminder':'home', 'reminder_update':'home', 'smart_home':'home', 
        'traffic':'auto', 'directions':'auto', 'gas':'auto', 'gas_type':'auto', 'distance':'auto', 'current_location':'auto', 'mpg':'auto', 'oil_change_when':'auto', 'oil_change_how':'auto', 'jump_start':'auto', 'uber':'auto', 'schedule_maintenance':'auto', 'last_maintenance':'auto', 'tire_pressure':'auto', 'tire_change':'auto', 
        'book_flight':'travel', 'book_hotel':'travel', 'car_rental':'travel', 'travel_suggestion':'travel', 'travel_alert':'travel', 'travel_notification':'travel', 'carry_on':'travel', 'timezone':'travel', 'vaccines':'travel', 'translate':'travel', 'flight_status':'travel', 'international_visa':'travel', 'lost_luggage':'travel', 'plug_type':'travel', 'exchange_rate':'travel', 
        'time':'utility', 'alarm':'utility', 'share_location':'utility', 'find_phone':'utility', 'weather':'utility', 'text':'utility', 'spelling':'utility', 'make_call':'utility', 'timer':'utility', 'date':'utility', 'calculator':'utility', 'measurement_conversion':'utility', 'flip_coin':'utility', 'roll_dice':'utility', 'definition':'utility', 
        'direct_deposit':'work', 'pto_request':'work', 'taxes':'work', 'payday':'work', 'w2':'work', 'pto_balance':'work', 'pto_request_status':'work', 'next_holiday':'work', 'insurance':'work', 'insurance_change':'work', 'schedule_meeting':'work', 'pto_used':'work', 'meeting_schedule':'work', 'rollover_401k':'work', 'income':'work', 
        'greeting':'talk', 'goodbye':'talk', 'tell_joke':'talk', 'where_are_you_from':'talk', 'how_old_are_you':'talk', 'what_is_your_name':'talk', 'who_made_you':'talk', 'thank_you':'talk', 'what_can_i_ask_you':'talk', 'what_are_your_hobbies':'talk', 'do_you_have_pets':'talk', 'are_you_a_bot':'talk', 'meaning_of_life':'talk', 'who_do_you_work_for':'talk', 'fun_fact':'talk', 
        'change_ai_name':'meta', 'change_user_name':'meta', 'cancel':'meta', 'user_name':'meta', 'reset_settings':'meta', 'whisper_mode':'meta', 'repeat':'meta', 'no':'meta', 'yes':'meta', 'maybe':'meta', 'change_language':'meta', 'change_accent':'meta', 'change_volume':'meta', 'change_speed':'meta', 'sync_device':'meta'}
        return label_dict[fine_label]
    
    def wos_fine_to_coarse(self, fine_label):
        """
        Convert fine labels to coarse ones for WOS dataset.
        """
        return fine_label.split('.')[0]

    def hwu_fine_to_coarse(self, fine_label):
        """
        Convert fine labels to coarse ones for HWU64 dataset.
        """
        return fine_label.split('_')[0]

class InputExample(object):
    """
    Convert data to examples for subsequent processing.
    """
    def __init__(self, text, label_coarse=None, label_fine=None):
        self.text = text
        self.label_coarse = label_coarse
        self.label_fine = label_fine

class InputFeatures(object):
    """
    Input features for BERT.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id_coarse, label_id_fine):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id_coarse = label_id_coarse
        self.label_id_fine = label_id_fine