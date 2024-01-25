import shutil
import os

folder_to_zip = 'LCFN/experiment_result'  # Replace 'your_folder' with the name of the folder you want to zip

shutil.make_archive(folder_to_zip, 'zip', folder_to_zip)

# Move the zip file to the current working directory
shutil.move(f'{folder_to_zip}.zip', f'./{folder_to_zip}.zip')

print(f'Zip file created: {folder_to_zip}.zip')