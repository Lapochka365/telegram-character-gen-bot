from bs4 import BeautifulSoup
import pandas as pd
import yaml


def clean_data(html_file):

    soup = BeautifulSoup(html_file, 'html.parser')
    all_messages_raw = soup.findAll(
        class_='im-mess--text wall_module _im_log_body'
    )
    all_messages_clean = []
    for message in all_messages_raw:
        all_messages_clean.append(message.text.strip())

    all_messages_clean = list(filter(bool, all_messages_clean))

    return all_messages_clean


if __name__ == '__main__':

    with open('config.yml', 'r') as f:
        PATHS = yaml.load(f, Loader=yaml.Loader)['html_vk_messages_paths']

    clean_messages = []
    for data_path in PATHS:
        with open(data_path, 'r') as f:
            html_page = f.read()
            result_messages = clean_data(html_page)
            clean_messages.extend(result_messages)

    messages_df = pd.DataFrame(clean_messages, columns=['sentences'])
    messages_df.to_csv(r'processed data\messages_dataset.csv', index=False)
    print('>>> Messages were parsed and put into .csv file')
