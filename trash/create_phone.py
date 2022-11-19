
def load_phone_mfa(filename):
    phone_to_id = dict()
    id_to_phone = dict()
    textfile = open("phone_only.txt", "w")
    phone_list = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            phone, id = line.strip().split("\t", 1)
            phone_list.append(phone)
            print(phone)
            phone_to_id[phone] = int(id)
            id_to_phone[int(id)] = phone
    phone_list.sort()    
    for element in phone_list:
        tmp = "{0}".format(element)
        textfile.write(tmp + "\n")
    return phone_to_id, id_to_phone

phone_to_id, id_to_phone = load_phone_mfa("/home/chungtran/Code/TTS/FastSpeech2/text_bak1/phones.sym")
