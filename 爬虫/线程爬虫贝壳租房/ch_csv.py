import csv
import random

data = ['house_id', 'collect_id', 'city_id', 'user_id', 'house_title', 'house_state', 'room_num', 'wc_num', 'room_area', 'deposit', 'price', 'address', 'house_desc', 'house_info', 'house_img']

with open('./data.csv', 'r', ) as f:
    file_read = csv.reader(f)
    header = file_read.__next__()
    with open('./new_data.csv', 'w', encoding='utf-8', newline='') as f_w:
        file_wirter = csv.writer(f_w)
        file_wirter.writerow(data)
        index = 1
        for line in file_read:
            new_data = [index, random.randint(1,20), random.randint(1,6), random.randint(1,10),
                        line[0], 0, random.randint(1, 4), random.randint(1, 2), int(line[5][:-1].split('.')[0]), random.choice([500, 1000, 800]),
                        int(line[8][:-3]), line[4], '+'.join([line[9], line[10], line[11]]), '+'.join([line[12], line[13], line[14], line[15]]), 'https://ke-image.ljcdn.com/110000-inspection/pc1_QIPKyj4d0.jpg!m_fill,w_250,h_182,l_fbk,o_auto']
            index += 1
            file_wirter.writerow(new_data)
        print('finished')

