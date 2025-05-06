def main():
    copy_directory = ['socket_client']
    source_directory = 'Separate_headers'
    copy_filenames = ['NN_enums.h', 'data_type.h']
    for file_name in copy_filenames:
        content = ''
        with open(f"{source_directory}/{file_name}", 'r') as file:
            content = file.read()
        for directory in copy_directory:
            with open(f"{directory}/{file_name}", 'w') as file:
                file.write(content)
    

if __name__ == '__main__':
    main()
