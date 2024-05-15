import shutil

# Create a zip archive of the directory '/home/dinithi/game_sessions' and save it as '/home/dinithi/game_sessions.zip'
archived = shutil.make_archive("/home/dinithi/game_sessions", 'zip', '/home/dinithi/game_sessions/')