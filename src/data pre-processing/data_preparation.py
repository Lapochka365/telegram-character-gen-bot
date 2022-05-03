import pandas as pd
import numpy as np
import pickle
import yaml


# delete unwanted words in messages inplace
def delete_unwanted_substrings(substrings_to_delete, dataframe_with_messages):

    for idx, message in enumerate(dataframe_with_messages['sentences']):
        for string_to_find in substrings_to_delete:
            if string_to_find in message:
                length_of_delete = len(string_to_find)
                index_of_substring = message.find(string_to_find)
                message_first_part = message[0:index_of_substring]
                message_second_part = \
                    message[index_of_substring+length_of_delete:]
                message = message_first_part + message_second_part
                dataframe_with_messages.at[idx, 'sentences'] = message


# change messages dataframe into python string with '! ' delimiter
def from_df_to_string(dataframe_with_messages):

    messages_to_string = str()
    for message in dataframe_with_messages['sentences']:
        messages_to_string += message.strip() + '! '

    return messages_to_string


# copy dataset onto itself in order to increase total number of samples
def copy_messages_dataset(messages_dataset):

    messages_dataset_copy = messages_dataset.copy(deep=True)
    messages_dataset_concat = pd.concat(
        [messages_dataset, messages_dataset_copy]
    ).reset_index(drop=True)

    return messages_dataset_concat


if __name__ == '__main__':

    with open('config.yml', 'r', encoding='utf-8') as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    PATH_TO_MESSAGES = config_file['messages_dataset_path']
    SUBSTRINGS_TO_DELETE = config_file['substrings_to_delete_in_messages']
    HOW_MANY_COPIES_OF_MESSAGES_DF = \
        int(config_file['how_many_times_to_copy_msg_df'])

    messages = pd.read_csv(PATH_TO_MESSAGES)

    delete_unwanted_substrings(SUBSTRINGS_TO_DELETE, messages)

    if HOW_MANY_COPIES_OF_MESSAGES_DF:
        for _ in range(HOW_MANY_COPIES_OF_MESSAGES_DF):
            messages = copy_messages_dataset(messages)

    text = from_df_to_string(messages)
    unique_chars = tuple(set(text))
    int2char = dict(enumerate(unique_chars))
    char2int = {char: index for index, char in int2char.items()}

    encoded_text = np.array([char2int[char] for char in text])

    with open(r'processed data\unique_chars.pkl', 'wb') as f:
        pickle.dump(unique_chars, f)

    with open(r'processed data\encoded_text.pkl', 'wb') as f_2:
        pickle.dump(encoded_text, f_2)

    if HOW_MANY_COPIES_OF_MESSAGES_DF:
        print('>>> Messages were copied onto itself, '
              'encoded and saved to 2 files:')
    else:
        print('>>> Messages were encoded and saved to 2 files:')
    print('>>> - unique_chars.pkl with unique characters')
    print('>>> - encoded_text.pkl with encoded messages converted to string')
