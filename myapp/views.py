from django.shortcuts import render, redirect
from django.http import HttpResponse
import tensorflow as tf
import tensorflow_text as text
import time
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from model.codet5model import CodeT5
import torch
from transformers import RobertaTokenizer,T5ForConditionalGeneration
ques='add 1 and 2'
output1 = '#include <iostream>\nusing namespace std; \n int main(){ \n cout<<1+2; \n return 0;\n}'
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

def home(request):
    return render(request, 'home.html')

def result(request):


    # print('--------------', tf.__version__)
    if request.method == "POST":
        pseudocode = request.POST.get('description')
        print('---------------------',pseudocode)
        # model = tf.saved_model.load('/home/rakshya/Documents/MajorProjectFrontend/PseudocodeModel/')
        # tokenizer([pseudocode])
        model = CodeT5()
        print(tokenizer([pseudocode]))
        torch.save(model.state_dict(), '/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth' )
        # input_tensor = torch.tensor(description)
        model.load_state_dict(torch.load('/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth'))
        # print(model.eval())
        input_ids = tokenizer(pseudocode , return_tensors='pt').input_ids
        attention_mask = tokenizer(pseudocode , return_tensors='pt').attention_mask
        model1 = T5ForConditionalGeneration.from_pretrained('/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/')
        outputs = model1.generate(input_ids,max_length=500)
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated output: ", predicted )

        # output_tensor = model(input_ids,attention_mask)
        # print(output_tensor)
        # output_data = {'output_data': output_tensor.tolist()}



        
        
        # predicted_output = model(description).numpy()

        # output=predicted_output.decode('utf-8')
        # print('--------------- ',output_data)
        # output1 = predicted_output.replace(';', ';\n') 
        # output1 = output1.replace('{', '{\n')
        # output1 = output1.replace(')', ')\n')
        # c=time.perf_counter()
        # print('Predicted Output: ',output)
        # print(result)
        # print('time to load model:  ',round(b-a)," sec")
        # print( 'time to generate & display output:  ', round(c-b)," sec\n")
        # return JsonResponse({'ques':description,'result': output})
        return render(request, 'result.html', {'ques':pseudocode,'result': predicted})
    # return JsonResponse({})




# def api_expose(request):
    # if request.method == "POST":
    #     description = request.POST.get('description')
    #     print('---------------------',description)
    #     model = tf.saved_model.load('/home/rakshya/Documents/MajorProjectFrontend/PseudocodeModel/')


        # Specify a path
        # PATH = "CodeT5Model3/pytorch_model.bin"
        # the_model = TheModelClass(*args, **kwargs)
        # the_model.load_state_dict(torch.load(PATH))
        # model = torch.load(PATH)
        # print("----------------------------------- " ,model)
        # model.eval()
        # predicted_output = model(description).numpy()

        # output=predicted_output.decode('utf-8')
    # return JsonResponse({'ques':"ques",'result': "output1"})

def api_expose(request):
    print("ques------------------", ques)
    print("out=------------------",output1)
    return JsonResponse({'ques':ques,'result': output1})
