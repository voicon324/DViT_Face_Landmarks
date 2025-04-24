import os
import argparse

def create_300w_filelist(root_data_dir, output_file, subfolders_to_scan):
    """
    Quét các thư mục con của 300W và tạo file list.

    Args:
        root_data_dir (str): Đường dẫn đến thư mục gốc chứa dữ liệu 300W
                             (ví dụ: '/path/to/300w_data').
        output_file (str): Đường dẫn đến file list text sẽ được tạo
                           (ví dụ: 'my_300w_train_list.txt').
        subfolders_to_scan (list): Danh sách các đường dẫn thư mục con (tương đối
                                   so với root_data_dir) cần quét để tìm file.
                                   Ví dụ: ['lfpw/trainset', 'helen/trainset', 'afw']
    """
    count = 0
    basenames_found = []

    print(f"Scanning folders in {root_data_dir}...")
    print(f"Folders to scan: {subfolders_to_scan}")

    for subfolder in subfolders_to_scan:
        current_folder_path = os.path.join(root_data_dir, subfolder)
        print(f"Scanning: {current_folder_path}")

        if not os.path.isdir(current_folder_path):
            print(f"Warning: Subfolder not found or is not a directory: {current_folder_path}. Skipping.")
            continue

        for filename in os.listdir(current_folder_path):
            basename, ext = os.path.splitext(filename)

            if ext.lower() in ['.png', '.jpg']:
                pts_filename = basename + '.pts'
                pts_filepath = os.path.join(current_folder_path, pts_filename)

                if os.path.exists(pts_filepath):
                    relative_basename = os.path.join(subfolder, basename)
                    relative_basename = relative_basename.replace('\\', '/')
                    basenames_found.append(relative_basename)
                    count += 1

    print(f"Found {count} valid image/landmark pairs.")

    if not basenames_found:
        print("Error: No valid pairs found. Please check root_data_dir and subfolders_to_scan.")
        return

    basenames_found.sort()

    print(f"Writing file list to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            for bn in basenames_found:
                f.write(bn + '\n')
        print("File list created successfully.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate file lists for 300W dataset.')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the root 300W dataset directory.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output list files.')
    args = parser.parse_args()

    ROOT_300W_DIRECTORY = args.root_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TRAIN_SUBFOLDERS = ['lfpw/trainset', 'helen/trainset', 'afw']
    TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'generated_300w_train_list.txt')

    TEST_SUBFOLDERS = ['lfpw/testset', 'helen/testset', 'ibug']
    TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'generated_300w_test_list.txt')

    if not os.path.isdir(ROOT_300W_DIRECTORY):
        print(f"Error: Root directory not found: {ROOT_300W_DIRECTORY}")
        return

    print("Generating Training List...")
    create_300w_filelist(ROOT_300W_DIRECTORY, TRAIN_OUTPUT_FILE, TRAIN_SUBFOLDERS)

    print("-" * 20)

    print("Generating Testing List...")
    create_300w_filelist(ROOT_300W_DIRECTORY, TEST_OUTPUT_FILE, TEST_SUBFOLDERS)

if __name__ == '__main__':
    main()