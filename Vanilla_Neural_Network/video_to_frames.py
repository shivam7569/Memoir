import cv2
import os

try:
    if not os.path.exists('../Data_Memoir'):
        os.mkdir('../Data_Memoir')
except OSError:
    print('Could not make the Data_Memoir directory.')

CURRENTFRAME = 0
folder = 'Real_1'

def process(folder, animation, currentframe):
    
    cam = cv2.VideoCapture('./Data/' + folder + '/' + animation + ".mp4") 
    
    while(True): 
    
        if currentframe == 50000:
            break
    
        ret, frame = cam.read() 
    
        if ret:
    
            os.chdir('/home/andy/Projects/Data_Memoir/')
    
            try:
                if not os.path.exists(folder):
                    os.mkdir(folder)
            except OSError:
                print('Could not make ' + folder + ' directory.')
            
            os.chdir('./' + folder + '/')
            
            try: 
                if not os.path.exists('Frames'): 
                    os.makedirs('Frames') 
            except OSError: 
                print ('Error: Creating directory of data') 
            
            name = './Frames/' + folder + '_frame_' + str(currentframe).zfill(5) + '.jpg'
            print ('Creating...' + name) 

            cv2.imwrite(name, frame) 
    
            currentframe += 1
        else: 
            break
    
    return currentframe

    cam.release() 
    cv2.destroyAllWindows() 

anim = folder
CURRENTFRAME = process(folder, anim, CURRENTFRAME)