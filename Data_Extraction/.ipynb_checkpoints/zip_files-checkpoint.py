import shutil
import os
# import io
import multiprocessing

root_dir="/data/VR_NET/folders"

output_dir="/data/VR_NET/game_sessions/"



# Creating the ZIP file 


def worker(game):
    # for game in games_to_zip:
    # print(game)
    try:
        file_to_zip=os.path.join(root_dir,game)
        path_to_store=os.path.join(output_dir,game)
        # print(file_to_zip,path_to_store)
        archived = shutil.make_archive(path_to_store, 'zip', file_to_zip)
    except:
        print(game)
        # continue
        
def main():
    # games_to_zip=os.listdir(root_dir)
    games_to_zip=['Metdonalds',
 'Venues',
 'Mars_Miners',
 'The_aquarium',
 'Arena_Clash',
 'Wake_the_Robot',
 'Kowloon',
 'Earth_Gym',
 'Super_Rumble',
 'Wild_Quest',
 'UFO_crash_site_venue',
 'American_Idol',
 'Army_Men',
 'Bobber_Bay_Fishing',
 'Citadel',
 '3D_Play_House',
 'Scifi_Sandbox',
 'Superhero_Arena']
    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, games_to_zip):
        count += 1
        print(count)

if __name__ == "__main__":
    main()