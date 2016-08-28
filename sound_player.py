import os
import pygame

pygame.mixer.init()

sound_dict = {'kitchen':'stove_and_faucet.ogg',
              'bathroom':'slippery_floor.ogg',
              'street_building':'remember_phone_etc.ogg'
              }

def play_sound(sound_key):
    pygame.mixer.music.load(os.path.join('./sound_clips', sound_dict[sound_key]))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    
