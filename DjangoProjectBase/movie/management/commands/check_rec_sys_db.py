from dotenv import load_dotenv, find_dotenv
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = 'Modify path of images'

    def handle(self, *args, **kwargs):

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']
        
        items = Movie.objects.all()

        req = "película de la segunda guerra mundial"
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        print(items[idx].title)