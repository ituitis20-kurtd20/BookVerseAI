import torch
from transformers import AutoTokenizer, AutoModel
from supabase import create_client
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# Initialize the model and tokenizer
model = AutoModel.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")
tokenizer = AutoTokenizer.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")

# Initialize Supabase client
url = "https://dujnhstimlhkodtayygi.supabase.co"

key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR1am5oc3RpbWxoa29kdGF5eWdpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE4NDI1NDgsImV4cCI6MjA0NzQxODU0OH0.8JpiFgmTtzwl1RS6xrz3npVog1XhjgqqhXQX6rvBvmE"
client = create_client(url, key)

def get_embedding(sentences, model, tokenizer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

@csrf_exempt

def semantic_search(request):
    if request.method == "GET":
        return render(request, "semantic_search.html")
    elif request.method == "POST":
        # Extract form data
        query = request.POST.get("query", "")
        match_threshold = float(request.POST.get("match_threshold", 0.7))
        match_count = int(request.POST.get("match_count", 10))

        try:
            # Process the data
            embedding = get_embedding(query, model, tokenizer).tolist()[0]
            response = client.rpc(
                "semantic_search",
                {
                    "query_embedding": embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                },
            ).execute()

            if response.data:
                return JsonResponse({"status": "success", "recommendations": response.data})
            else:
                return JsonResponse({"status": "error", "message": "Cannot find similar results."})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def recommend_books(request):
    if request.method == "GET":
        return render(request, "recommend_books.html")
    elif request.method == "POST":
        # Convert form data to JSON-like structure
        user_id = int(request.POST.get("user_id", 0))
        top_n = int(request.POST.get("top_n", 10))
        similarity_threshold = float(request.POST.get("similarity_threshold", 0.8))

        try:
            # Process the data as before
            response = client.rpc(
                "recommend_books",
                {
                    "get_user_id": user_id,
                    "top_n": top_n,
                    "similarity_threshold": similarity_threshold,
                },
            ).execute()

            if response.data:
                return JsonResponse({"status": "success", "recommendations": response.data})
            else:
                return JsonResponse({"status": "error", "message": "Cannot find similar results."})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
