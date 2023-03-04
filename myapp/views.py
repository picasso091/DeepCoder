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
from transformers import RobertaTokenizer, T5ForConditionalGeneration
ques = '''print hello world'''
output_t5 = '#include <iostream> \n using namespace std; \n int main(){ \n cout<<"hello world"<<endl; \n}'
output_vanilla = '#include <iostream> \n using namespace std; \n int main(){ \n cout<<"karen"<<endl; \n return 0; \n}'


def home(request):
    return render(request, 'home.html')

def result(request):
    # print('--------------', tf.__version__)
        if 'vanilla' in request.POST:
            try:
                text = request.POST.get('description')
                print('text-vanilla  ', text)
                a = time.perf_counter()       
                model_vanilla = tf.saved_model.load(
                    '/home/rakshya/Documents/MajorProjectFrontend/VanillaModel/')

                output_vanilla = model_vanilla(text).numpy()
                predicted_vanilla = output_vanilla.decode('utf-8')

                if (';' in predicted_vanilla):
                    predicted_vanilla = predicted_vanilla.replace(';', ';\n')
                elif (')' in predicted_vanilla):
                    predicted_vanilla = predicted_vanilla.replace(')', ')\n')
                elif ('{' in predicted_vanilla):
                    predicted_vanilla = predicted_vanilla.replace('{', '{\n')
                elif ('}' in predicted_vanilla):
                    predicted_vanilla = predicted_vanilla.replace('}', '}\n')
                else:
                    predicted_vanilla = predicted_vanilla.replace('>', '>\n')
                    print('Predicted Output: ', predicted_vanilla)
                b = time.perf_counter()
                # print("Generated output: ", predicted )

                print('vanilla  - time to generate & display output:  ',
                    round(b-a), " seconds\n")
                return render(request, 'result_vanilla.html', {'ques': text,'result_vanilla': predicted_vanilla})
            except ValueError:
                return 'result_vanilla.html'
            
        elif 'codet5' in request.POST:
            try:
                text = request.POST.get('description')
                print('t5 ', text)
                a = time.perf_counter()

                model = CodeT5()
                # print(tokenizer([text]))
                torch.save(model.state_dict(
                ), '/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth')
                # input_tensor = torch.tensor(description)
                # model.load_state_dict(torch.load('/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth'))
                # print(model.eval())
                tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
                input_ids = tokenizer(text, return_tensors='pt').input_ids
                attention_mask = tokenizer(text, return_tensors='pt').attention_mask
                model_t5 = T5ForConditionalGeneration.from_pretrained(
                    '/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/')
                # print(model_t5.eval())
                output_t5 = model_t5.generate(input_ids, max_length=500)
                predicted_t5 = tokenizer.decode(output_t5[0], skip_special_tokens=True)

                b = time.perf_counter()

                print('t5 - time to generate & display output:  ', round(b-a), " sec\n")


                return render(request, 'result_t5.html', {'ques': text, 'result_t5': predicted_t5})
            except ValueError:
                return 'result_t5.html'
# def result_t5(request):

#     # print('--------------', tf.__version__)
#     if request.method == "POST":
#         text = request.POST.get('description')
#         print('t5 ', text)
#         a = time.perf_counter()

#         model = CodeT5()
#         # print(tokenizer([text]))
#         torch.save(model.state_dict(
#         ), '/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth')
#         # input_tensor = torch.tensor(description)
#         # model.load_state_dict(torch.load('/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/model_weights.pth'))
#         # print(model.eval())
#         tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
#         input_ids = tokenizer(text, return_tensors='pt').input_ids
#         attention_mask = tokenizer(text, return_tensors='pt').attention_mask
#         model_t5 = T5ForConditionalGeneration.from_pretrained(
#             '/home/rakshya/Documents/MajorProjectFrontend/CodeT5Model/')
#         # print(model_t5.eval())
#         output_t5 = model_t5.generate(input_ids, max_length=500)
#         predicted_t5 = tokenizer.decode(output_t5[0], skip_special_tokens=True)

#         b = time.perf_counter()

#         print('t5 - time to generate & display output:  ', round(b-a), " sec\n")

#         # return JsonResponse({'ques':description,'result': output})
#         return render(request, 'result_t5.html', {'ques': text, 'result_t5': predicted_t5})


def api_expose(request):
    return JsonResponse({'pseudocode': ques, 'result_t5': output_t5, 'result_vanilla': output_vanilla})
