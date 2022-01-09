from django.shortcuts import render, redirect,get_object_or_404

from django.http import HttpResponseRedirect,HttpResponse
from django.shortcuts import render
from django.template import RequestContext
from .forms import DocumentForm
from .models import Contract,Question
import pdfminer
from django.core.files.storage import FileSystemStorage
from django.core.exceptions import SuspiciousOperation

import pdfminer.high_level
from pdfminer.layout import LAParams

from django.utils.timezone import now
from django.views.generic import DetailView

import fitz
import numpy as np
import os


colors = [ 'background-color:#ff8080', 'background-color:#ff9494', 'background-color:#ffa8a8', 'background-color:#ffbdbd',
'background-color:#df9f9f', 'background-color:#e4afaf', 'background-color:#e9bebe', 'background-color:#eecdcd',
'background-color:#ffdf80', 'background-color:#ffe494', 'background-color:#ffe9a8', 'background-color:#ffeebd',
'background-color:#dfcf9f', 'background-color:#e4d7af', 'background-color:#e9debe', 'background-color:#eee6cd',
'background-color:#bfff80', 'background-color:#c9ff94', 'background-color:#d4ffa8', 'background-color:#deffbd',
'background-color:#bfdf9f', 'background-color:#c9e4af', 'background-color:#d4e9be', 'background-color:#deeecd',
'background-color:#80ff9f', 'background-color:#94ffaf', 'background-color:#a8ffbe', 'background-color:#bdffcd',
'background-color:#9fdfaf', 'background-color:#afe4bc', 'background-color:#bee9c9', 'background-color:#cdeed6',
'background-color:#80ffff', 'background-color:#94ffff', 'background-color:#a8ffff', 'background-color:#bdffff',
'background-color:#9fdfdf', 'background-color:#afe4e4', 'background-color:#bee9e9', 'background-color:#cdeeee',
'background-color:#809fff', 'background-color:#94afff', 'background-color:#a8beff', 'background-color:#bdcdff',
'background-color:#9fafdf', 'background-color:#afbce4', 'background-color:#bec9e9', 'background-color:#cdd6ee',
'background-color:#bf80ff', 'background-color:#c994ff', 'background-color:#d4a8ff', 'background-color:#debdff',
'background-color:#bf9fdf', 'background-color:#c9afe4', 'background-color:#d4bee9', 'background-color:#decdee',
'background-color:#ff80df', 'background-color:#ff94e4', 'background-color:#ffa8e9', 'background-color:#ffbdee',
'background-color:#df9fcf', 'background-color:#e4afd7', 'background-color:#e9bede', 'background-color:#eecde6', ]



def model_form_upload(request):

    THIS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspdfpath = os.path.join(THIS_FOLDER, 'output.pdf')
    if request.method == 'POST':
        pdfname = "../output.pdf"
     
        questions = Question.objects.all()
        if request.session.has_key('start'):
            return render(request, 'pdfview.html', {'questions': questions, 'pdfname': pdfname})
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            contract = form.save(commit=False)
            try:
                os.remove(abspdfpath)
            except OSError:
                pass
            scanflag = contract.scanned
            contract.pdf.save('output.pdf', contract.pdf)
            if not scanflag:

                text = gettext(abspdfpath)
            else:
                from .pdftextextract import getscannedtext
                getscannedtext(abspdfpath)
                text = gettext(abspdfpath)

            if len(text)<=30:
                error = "The file you uploaded contains little or no text to review. Please read the instructions and try again."
                return render(request, 'index.html', {
                    'form': form, 'error':error
                })
            request.session['extracted_text'] = text
            request.session['start'] = True
            done_ques = [False]*41
          
            request.session['done_ques'] = done_ques
            return render(request, 'pdfview.html',{'questions':questions,'pdfname':'../output.pdf'})
         


    else:
        try:
            os.remove(abspdfpath)
        except OSError:
            pass
        request.session.flush()
        form = DocumentForm()
    return render(request, 'index.html', {
        'form': form
    })



def gettext(pdf):
    #print(pdf)
    lp = LAParams(all_texts=True)
    a= pdfminer.high_level.extract_text(pdf,laparams=lp)

    a.replace('\n',' ')
    a.replace('\n',' ')
    #print(a)
    return a

def gettext2(pdf):
    text = ''
    with fitz.open(pdf) as doc:
        for page in doc:
            text+= page.getText()
    return text


import matplotlib.colors

def pdfhighlight(answer,color,desc,alias):
    ### READ IN PDF
    THIS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspdfpath = os.path.join(THIS_FOLDER, 'output.pdf')
    doc = fitz.open(abspdfpath)
    if len(answer)<3:
        return
    for page in doc:
        ### SEARCH
        text = "Agreement"
        text_instances = page.searchFor(answer)

        ### HIGHLIGHT
        for inst in text_instances:
            highlight = page.add_rect_annot(inst)
            hex = color
            h = hex.lstrip('#')
            abc = matplotlib.colors.to_rgb(hex)
            h2 = page.addFreetextAnnot(inst,text=alias+": "+desc)  #add ques desc here
            h2.update(opacity=0)
            highlight.set_colors(fill=abc)
            highlight.update(opacity=0.3)
           # print(rgb)


    ### OUTPUT
   # doc.save("output.pdf", garbage=4, deflate=True, clean=True)
    doc.save(doc.name, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)

from django.shortcuts import render
from django.http import FileResponse, Http404



def outpdf(request):
    THIS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspdfpath = os.path.join(THIS_FOLDER, 'output.pdf')
    try:
        return FileResponse(open(abspdfpath, 'rb'), content_type='application/pdf')
    except FileNotFoundError:
        raise Http404('not found')

def pdfview(request):
    return render(request,'hhtml.html')
def test(request):
    return render(request,'pdfview.html')

def question_highlight(request,pk):
    question = get_object_or_404(Question,pk=pk)
    return render(request,'evaluation_results.html',{'pk':pk})


def evaluation_results(request,pk):
    import time
    timeprefix = "Time for Latest Inference:\n"
    #request.session.set_expiry(0)
    if 'done_ques' in request.session:
        if request.session.has_key('extracted_text'):
            done_ques = request.session['done_ques']
         
            question = get_object_or_404(Question, pk=pk)
            questions = Question.objects.all()
            ques = question.ques
            pdfname = "../output.pdf"
            if not done_ques[pk-2]:
                done_ques[pk-2] = True
            
                request.session['done_ques'] = done_ques
                start = time.time()
                text2=request.session['extracted_text']
                result = "hello"
               
                result = qa_model_inference(question=ques, context=text2, request=request)
                restext = result['prediction_text']
                for res in restext:
                    pdfhighlight(res['text'],question.color,question.description,question.alias)
           
                timetaken =  time.time() - start
           
                return render(request, 'pdfview.html', {'result': result,'pdfname':pdfname,'questions':questions,'timetaken':timeprefix+str(timetaken)+' seconds'})
            else:
                return render(request, 'pdfview.html', {'pdfname':pdfname,'questions':questions,'timetaken':timeprefix+'0 seconds : Clause Already Highlighted'})


        else:
            no_text_error = "Failed to retrieve any text from the document. Please try again!"
            return render(request, 'evaluation_results.html',
                          {'no_text_error': no_text_error})

    else:
        raise SuspiciousOperation


def qa_model_inference(question, context, request):
    import os
    import math
    import torch
    import transformers
    import collections
    from tqdm.auto import tqdm
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer

    # Preprocessing data to convert in qa form to feed to the model
    sample = {'context': [context],
              'question': [question],
              'id': ['1']}

    samples = Dataset.from_dict(sample)
    THIS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspdfpath = os.path.join(THIS_FOLDER, 'E4_roberta-base/')
    # load model from local directory
    model_directory = abspdfpath
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    model = AutoModelForQuestionAnswering.from_pretrained(model_directory, local_files_only=True)

    #quantize model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)

    # load tensors to cpu for quantization if not loaded already
    quantized_model = quantized_model.to(torch.device('cpu'))

    # def print_size_of_model(model):
    #     torch.save(model.state_dict(), "temp.p")
    #     print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    #     os.remove('temp.p')
    #
    # print_size_of_model(model)
    # print_size_of_model(quantized_model)

    trainer = Trainer(quantized_model)

    # setting model hyperparameters
    max_length = 384
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"
    n_best_size = 20
    max_answer_length = 384

    # function to convert data to features
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # calling function to convert to features
    validation_features = samples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=samples.column_names
    )

    # predicting answers
    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"],
                                   columns=list(validation_features.features.keys()))

    examples = samples
    features = validation_features

    # mapping example index to its corresponding features indices
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # function to postprocess qa predictions
    def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20,
                                   max_answer_length=max_answer_length):

        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        # predictions = collections.OrderedDict() # for single prediction
        predictions = collections.defaultdict(list)  # for more than one prediction

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                # best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0] # for single prediction
                list_of_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0:20]
                best_answer = []
                best_answer.append(list_of_answers[0])
                a = 1
                while (list_of_answers[a]['score'] >= (list_of_answers[0]['score'] - 1.0)):
                    best_answer.append(list_of_answers[a])
                    a = a + 1
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            # if not squad_v2:
            #     predictions[example["id"]] = best_answer["text"]
            # else:
            # answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            # predictions[example["id"]] = answer
            # predictions[example["id"]] = best_answer["text"] # for single predicition
            k = 0
            if a == 1:
                predictions[example['id']].append(
                    best_answer[0])  # for only text and not score .append(best_answer[0]['text'])
            else:
                while (k < a):
                    predictions[example['id']].append(best_answer[k])
                    k = k + 1

        return predictions

    # calling postprocessing function
    final_predictions = postprocess_qa_predictions(samples, validation_features, raw_predictions.predictions)

    # formatting answers properly
    formatted_predictions = [{"question": question, "prediction_text": v} for k, v in final_predictions.items()]

    return formatted_predictions[0]
