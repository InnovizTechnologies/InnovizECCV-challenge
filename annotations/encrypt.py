from pathlib import Path
from cryptography.fernet import Fernet

def main():
    key = (Path(__file__).parent.parent / 'evaluation_script' / 'key.txt').read_bytes()
    file = Path(__file__).parent / 'innoviz_2022-09-23_eval_gt.zip'
    data = file.read_bytes()

    # encrypt .zip file
    f = Fernet(key)
    data_enc = f.encrypt(data)

    # dump encrypted zip
    with open(f'{file}.enc', 'wb') as f:
        f.write(data_enc)
    

if __name__ == "__main__":
    main()