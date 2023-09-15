import os
import io
import cv2
import numpy as np
# import base64
import threading
from PIL import Image
from django.conf import settings
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required

# Create your views here.
global input_image
global image
@login_required(login_url='login')
def HomePage(request):
    return render (request,'home.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')
        


    return render (request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')

def LogoutPage(request):

    image_path = request.session.get('image_path')
    if image_path:
        os.remove(image_path)
        del request.session['image_path']
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.png')
    if os.path.exists(processed_image_path):
        os.remove(processed_image_path)

    logout(request)
    return redirect('login')


def select_image(request):

    if 'image_url' in request.session:
        del request.session['image_url']
    if 'image_path' in request.session:
        del request.session['image_path']
    if request.method == 'POST' and request.FILES.get('image'):
        # get the selected image and save it to the media folder
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'input_image.png')
        with open(image_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        # save the image URL to the session for display in the template
        image_url = os.path.join(settings.MEDIA_URL, 'input_image.png')
        request.session['image_url'] = image_url
        request.session['image_path'] = image_path

    return render(request, 'select_image.html')
    

def process_image(request):

    def process(image):

        quantization = 16
        tsimilarity = 5 # euclid distance similarity threshold
        tdistance =20 # euclid distance between pixels threshold
        vector_limit = 20 # shift vector elimination limit
        block_counter = 0
        block_size = 6
        
        image=cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        temp = []

        kernel = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 2, 1, 0],
                        [1, 2, -16, 2, 1],
                        [0, 1, 2, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.float32)

        # Normalize kernel
        kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

        # Apply LoG kernel to image
        filtered_image = cv2.filter2D(gray, -1, kernel)

        # Scale filtered image to 0-255 range and convert to uint8 datatype
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        arr = np.array(filtered_image)
        prediction_mask = np.zeros((arr.shape[0], arr.shape[1]))
        column = arr.shape[1] - block_size
        row = arr.shape[0] - block_size
        dcts = np.empty((((column+1)*(row+1)), quantization+2))
        
        for i in range(0, row):
            for j in range(0, column):

                blocks = arr[i:i+block_size, j:j+block_size]
                imf = np.float32(blocks) / 255.0  # float conversion/scale0
                dst = cv2.dct(imf)  # the dct
                blocks = np.uint8(np.float32(dst) * 255.0 ) # convert back
                # zigzag scan
                solution = [[] for k in range(block_size + block_size - 1)]
                for k in range(block_size):
                    for l in range(block_size):
                        sum = k + l
                        if (sum % 2 == 0):
                            # add at beginning
                            solution[sum].insert(0, blocks[k][l])
                        else:
                            # add at end of the list
                            solution[sum].append(blocks[k][l])

                for item in range(0,(block_size*2-1)):
                    temp += solution[item]

                temp = np.asarray(temp, dtype=float)
                temp = np.array(temp[:16])
                temp = np.floor(temp/quantization)
                temp = np.append(temp, [i, j])

                np.copyto(dcts[block_counter], temp)

                block_counter += 1
                temp = []



        dcts = dcts[~np.all(dcts == 0, axis=1)]
        dcts = dcts[np.lexsort(np.rot90(dcts))]



        sim_array = []
        for i in range(0, block_counter):
            if i <= block_counter-10:
                for j in range(i+1, i+10):
                    pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
                    pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
                    if pixelsim <= tsimilarity and pointdis >= tdistance:
                        sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])
            else:
                for j in range(i+1, block_counter):
                    pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
                    pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
                    if pixelsim <= tsimilarity and pointdis >= tdistance:
                        sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])



        sim_array = np.array(sim_array)
        delete_vec = []
        vector_counter = 0
        for i in range(0, sim_array.shape[0]):
            for j in range(1, sim_array.shape[0]):
                if sim_array[i][4] == sim_array[j][4] and sim_array[i][5] == sim_array[j][5] and vector_counter<=vector_limit:
                    vector_counter += 1
            if vector_counter < vector_limit:
                delete_vec.append(sim_array[i])
            vector_counter = 0

        delete_vec = np.array(delete_vec)
        delete_vec = delete_vec[~np.all(delete_vec == 0, axis=1)]
        delete_vec = delete_vec[np.lexsort(np.rot90(delete_vec))]

        for item in delete_vec:
            indexes = np.where(sim_array == item)
            unique, counts = np.unique(indexes[0], return_counts=True)
            for i in range(0, unique.shape[0]):
                if counts[i] == 6:
                    sim_array = np.delete(sim_array,unique[i],axis=0)

        
        for i in range(len(sim_array)):
            index1 = int(sim_array[i][0])
            index2 = int(sim_array[i][1])
            index3 = int(sim_array[i][2])
            index4 = int(sim_array[i][3])
            for j in range(0,7):
                for k in range(0,7):
                    prediction_mask[index1+j][index2+k] = 255
                    prediction_mask[index3+j][index4+k] = 255

        return Image.fromarray(prediction_mask)
    
    image_path = request.session.get('image_path')
    if not image_path:
        return HttpResponse('No image selected. Please select an image first.')

    image_url = request.session.get('image_url')

    # open the image and convert to PNG format
    with Image.open(image_path) as img:
        if img.format != 'PNG':
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img = Image.open(buffer)
        else :
            img = image_path

        # get the URLs for the images
        original_image_url = image_url
        processed_image_url = None

        # set a message to display while the image is being processed
        message = 'Image is being processed...'

        # save the processed image in the background
        # you can use Celery or any other task queue for this
        # here we use a simple threading approach
        
        def process_and_save_image():
            nonlocal processed_image_url
            nonlocal message
            try:
                processed_img = process(img).convert('L')
                processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.png')
                processed_img.save(processed_image_path)
                processed_image_url = os.path.join(settings.MEDIA_URL, 'processed_image.png')
                message = ''
            except Exception:
                message = 'Error occurred while processing the image.'
                raise

        t = threading.Thread(target=process_and_save_image)
        t.start()
        t.join()  # wait for the thread to finish processing the image

    return render(request, 'process_image.html', {
        'original_image_url': original_image_url,
        'processed_image_url': processed_image_url,
        'message': message,
    })
    
# ADDING DOWNLOAD OPTION
def download_image(request):
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.png')
    with open(processed_image_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="processed_image.png"'
        return response
