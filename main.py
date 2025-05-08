from collections import Counter, defaultdict
import os
import glob
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
import hdbscan
import dask.dataframe as dd  # Use Dask instead of pandas for scalable DataFrames
import matplotlib.pyplot as plt  # Use matplotlib for static plots
import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def classify_medical_outcomes(label_to_text: dict, model="gpt-4") -> dict:
    """
    Classifies each medical outcome as 'good', 'bad', or 'neutral' using GPT.
    From each cluster we select the most representative text and ask GPT to classify it.
    * this requires relatively very few calls to GPT
    """
    instruction = (
        "You are a medical assistant. Classify each of the following cases as 'good', 'bad', or 'neutral' outcome.\n"
        "A 'good' outcome means the treatment helped, or the patient is recovering.\n"
        "A 'bad' outcome means the condition worsened or the patient died.\n"
        "If unclear or balanced, choose 'neutral'.\n"
        "Return ONLY a JSON object with keys: 'good', 'bad', 'neutral', and values are lists of label names.\n\n"
    )
    case_lines = [f"{label}: {text}" for label, text in label_to_text.items()]
    full_prompt = instruction + "\n".join(case_lines)

    client = openai.OpenAI()  # requires OPENAI_API_KEY in environment
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0
        )
    except Exception as e:
        print(f"OpenAI API request failed: {e}")
        return {"good": [], "bad": [], "neutral": []}
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        for key in ['good', 'bad', 'neutral']:
            result.setdefault(key, [])
        return result
    except json.JSONDecodeError:
        # On parse failure, return neutral for all
        print("Failed to parse GPT output as JSON.")
        return {"good": [], "bad": [], "neutral": []}


def plot_doctor_cluster_frequencies(df, cluster_col, doctor_col='doctor_id', cluster_labels=None):
    """
    Plots the frequency of each doctor_id across a certain cluster_col,
    replacing cluster numbers with representative texts if provided.

    Args:
        df (pd.DataFrame): DataFrame containing doctor and cluster columns.
        cluster_col (str): Name of the cluster column.
        doctor_col (str): Name of the doctor ID column.
        cluster_labels (dict): Optional. A mapping from cluster number to representative text.
    """
    # Create a frequency table (cross-tabulation)
    freq_table = pd.crosstab(df[doctor_col], df[cluster_col])

    # Optionally rename cluster columns using representative texts
    if cluster_labels:
        freq_table = freq_table.rename(columns=cluster_labels)

    # Plot
    ax = freq_table.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Frequency of Consultation Clusters per Doctor")
    plt.xlabel("Doctor ID")
    plt.ylabel("Number of Consultations")
    plt.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    # Save using cluster_col name (remove file-invalid characters if needed)
    safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in cluster_col)
    plt.savefig(f"{safe_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def cluster_texts(texts, model_name='sentence-transformers/all-MiniLM-L6-v2', min_cluster_size=2):
    """
    Clusters a list of short texts using sentence embeddings and HDBSCAN.
    Returns a tuple of (cluster_label_to_text_dict, cluster_labels_list).
    """
    # Load pre-trained embedding model and compute embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    # Cluster using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    # Group texts by cluster label (excluding outliers labeled -1)
    cluster_texts_map = defaultdict(list)
    for label, text in zip(labels, texts):
        if label != -1:
            cluster_texts_map[label].append(text)
    # Representative text for each cluster (most frequent text in cluster)
    representative_texts = {
        label: Counter(texts_in_cluster).most_common(1)[0][0]
        for label, texts_in_cluster in cluster_texts_map.items()
    }
    return representative_texts, labels.tolist()


def structure_summary_with_llm(summary: str) -> list:
    """
    Extracts structured information from a medical summary using an LLM (GPT).
    Returns a list of four fields: [reason_for_consultation, medical_diagnosis, prescription, suggested_follow_up].
    """
    # Define expected output schema for the LLM
    response_schemas = [
        ResponseSchema(name="reason for consultation",
                       description="Why did the patient come (reported symptoms or reason)"),
        ResponseSchema(name="medical diagnosis", description="Diagnosed medical condition, if any"),
        ResponseSchema(name="prescription", description="Procedures, medications or treatments prescribed"),
        ResponseSchema(name="suggested follow up", description="Follow-up actions or timeline suggested")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # Few-shot examples added to system prompt to guide the model
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical assistant. Extract structured information from intake notes. "
         "If a field is not present in the text, return 'none'. Do not guess."
         "\nExamples:\n"
         "1. 'Discussed test results for CT scan. Findings indicate normal limits. Next steps: schedule a follow-up'.\n"
         "expected output:\n"
         "[reason for consultation: 'discuss test results', medical diagnosis: 'test shows normal limits', prescription: 'none', suggested follow up: 'schedule a follow-up']\n"
         "2. 'Follow-up for migraines. Improvements noted. Will reassess in 3 days.'.\n"
         "expected output:\n"
         "[reason for consultation: 'follow-up for migraines', medical diagnosis: 'shows improvement', prescription: 'none', suggested follow up: 'reassess in 3 days']\n"
         "3. 'Complaints of rash. Suspected bronchitis. Ordered blood test and started topical cream.'\n"
         "expected output:\n"
         "[reason for consultation: 'complaints of rash', medical diagnosis: 'bronchitis', prescription: 'blood test, topical cream', suggested follow up: 'none']\n"
         ),
        ("user", "{input}")
    ])
    chain_input = prompt.format_messages(input=summary + "\n\n" + parser.get_format_instructions())
    llm = ChatOpenAI(temperature=0)
    try:
        response = llm(chain_input)
    except Exception as e:
        # API call failed
        print(f"OpenAI API call failed for summary: {e}")
        return ["none", "none", "none", "none"]
    try:
        parsed = parser.parse(response.content)
    except Exception as e:
        # Parsing failed
        print("Failed to parse LLM output for summary, using 'none' for all fields.")
        return ["none", "none", "none", "none"]
    # Return the structured fields as a list
    return [
        parsed.get("reason for consultation", "none"),
        parsed.get("medical diagnosis", "none"),
        parsed.get("prescription", "none"),
        parsed.get("suggested follow up", "none")
    ]


# Helper function to process a partition of the DataFrame in parallel
def process_partition(partition):
    """
    Apply structure_summary_with_llm to each summary in the partition in parallel.
    Returns a pandas DataFrame with the structured fields.
    """
    columns = ['reason_for_consultation', 'medical_diagnosis', 'prescription', 'suggested_follow_up']

    # Batch process summaries to avoid too many threads
    def process_batch(batch):
        results = []
        for summary in batch:
            try:
                results.append(structure_summary_with_llm(summary))
            except Exception as e:
                results.append(["none", "none", "none", "none"])
        return results

    n = len(partition)
    results = [None] * n
    batch_size = 100  # number of summaries per task (tune for performance)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_index = {}
        for start in range(0, n, batch_size):
            batch = partition['summary'].iloc[start:start + batch_size].tolist()
            future = executor.submit(process_batch, batch)
            future_to_index[future] = start
        for future in as_completed(future_to_index):
            start_idx = future_to_index[future]
            try:
                batch_result = future.result()
            except Exception as e:
                # If a batch fails, fill its range with "none"
                batch_len = min(batch_size, n - start_idx)
                batch_result = [["none", "none", "none", "none"]] * batch_len
                print(f"Batch starting at index {start_idx} failed: {e}")
            # Store results in the correct positions
            results[start_idx:start_idx + len(batch_result)] = batch_result
    return pd.DataFrame(results, columns=columns)


def get_data(file_name):
    """
    Reads the input data file into a DataFrame (using Dask for scalability).
    """
    if file_name.endswith('.json'):
        # Use Dask to read large JSON (assuming JSON Lines format for scalability)
        try:
            data = dd.read_json(file_name, orient='records', lines=True, blocksize="64MB")
        except ValueError:
            # Fallback: read entire file if not line-delimited JSON
            data = dd.read_json(file_name, orient='records', lines=False, blocksize=None)
    elif file_name.endswith('.csv'):
        data = dd.read_csv(file_name)
    else:
        # Default to pandas for unknown small file formats
        data = pd.read_json(file_name)
    return data


def get_structured_summaries(df, file_name='structured_summaries-0.csv'):
    """
    Converts a DataFrame of medical summaries into structured form (reason, diagnosis, prescription, follow-up).
    Uses parallel LLM calls and writes results to CSV. Returns a DataFrame of structured summaries.
    """
    columns = ['reason_for_consultation', 'medical_diagnosis', 'prescription', 'suggested_follow_up']
    if isinstance(df, dd.DataFrame):
        # Distributed processing of summaries
        meta = {col: 'object' for col in columns}
        results_ddf = df.map_partitions(process_partition, meta=meta)
        # Save to CSV (one file per partition for scalability)
        base = os.path.splitext(file_name)[0]
        output_pattern = base + "-*.csv"
        results_ddf.to_csv(output_pattern, index=False)
        # Return a Dask DataFrame reading the saved CSV parts
        return dd.read_csv(output_pattern)
    else:
        # Fallback for small DataFrame: use sequential processing
        structured_records = []
        for summary in df['summary']:
            try:
                structured_records.append(structure_summary_with_llm(summary))
            except Exception as e:
                structured_records.append(["none", "none", "none", "none"])
        df_structured = pd.DataFrame(structured_records, columns=columns)
        df_structured.to_csv(file_name, index=False)
        return df_structured


def cluster_summaries_and_outcomes(df):
    """
    Clusters the text fields in the structured data (reason, diagnosis, prescription, follow-up, outcome).
    Returns the cluster-to-text mapping for future outcomes and the updated DataFrame with cluster labels.
    Creates visualizations of doctor performance based on clusters.
    FUTURE: current algo allows for several clusters which should be merged ("none" for instance). merge these together
    """
    # If df is distributed, convert to pandas for clustering (this step is memory-intensive for very large data)
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    # Perform clustering for each text column
    consultation_cluster_dict, consultation_labels = cluster_texts(df['reason_for_consultation'].tolist(),
                                                                   min_cluster_size=5)
    df['reason_for_consultation_clusters'] = consultation_labels
    plot_doctor_cluster_frequencies(df, cluster_col='reason_for_consultation_clusters',
                                    cluster_labels=consultation_cluster_dict)

    diagnosis_cluster_dict, diagnosis_labels = cluster_texts(df['medical_diagnosis'].tolist(), min_cluster_size=5)
    df['medical_diagnosis_clusters'] = diagnosis_labels
    plot_doctor_cluster_frequencies(df, cluster_col='medical_diagnosis_clusters', cluster_labels=diagnosis_cluster_dict)

    prescription_cluster_dict, prescription_labels = cluster_texts(df['prescription'].tolist(), min_cluster_size=5)
    df['prescription_clusters'] = prescription_labels
    plot_doctor_cluster_frequencies(df, cluster_col='prescription_clusters', cluster_labels=prescription_cluster_dict)

    follow_up_cluster_dict, follow_up_labels = cluster_texts(df['suggested_follow_up'].tolist(), min_cluster_size=5)
    df['suggested_follow_up_clusters'] = follow_up_labels
    plot_doctor_cluster_frequencies(df, cluster_col='suggested_follow_up_clusters',
                                    cluster_labels=follow_up_cluster_dict)

    outcomes_cluster_dict, outcomes_labels = cluster_texts(df['future_outcome'].tolist(), min_cluster_size=5)
    df['outcomes_clusters'] = outcomes_labels
    plot_doctor_cluster_frequencies(df, cluster_col='outcomes_clusters', cluster_labels=outcomes_cluster_dict)
    return outcomes_cluster_dict, df  # return mapping and updated DataFrame


def rank_doctors(df):
    """
    Ranks doctors based on the classification of their future outcomes.
    Creates pie charts for each doctor showing the distribution of outcome classifications.
    grade is calculated as follows:
    * good = +1
    * neutral = 0
    * bad = -1
    * rank = sum(good, neutral, bad)
    """
    cluster_to_class = {}
    for cls, clusters in future_outcome_classification.items():
        for cluster in clusters:
            cluster_to_class[int(cluster)] = cls

    # Step 2: Map to classification in the DataFrame
    df['class'] = df['outcomes_clusters'].map(cluster_to_class)

    # Step 3: Count class distribution per doctor
    doctor_class_counts = df.groupby(['doctor_id', 'class']).size().unstack(fill_value=0)

    # Step 4: Plot a pie chart for each doctor
    for doctor in doctor_class_counts.index:
        counts = doctor_class_counts.loc[doctor]
        counts = counts[counts > 0]  # remove 0 slices
        plt.figure()
        counts.plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Outcome Classification for {doctor}')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f"{doctor}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Step 5: Compute success score: good = +1, neutral = 0, bad = -1
    score_weights = {'good': 1, 'neutral': 0, 'bad': -1}
    score_df = doctor_class_counts.fillna(0).copy()
    score_df['success_score'] = sum(score_df.get(cls, 0) * weight for cls, weight in score_weights.items())

    # Step 6: Rank doctors
    score_df = score_df.sort_values(by='success_score', ascending=False)
    print("Doctor ranking by success score:")
    print(score_df[['success_score']])
    return


def prepare_data(df):
    # df = df[df['class'].isin(['good', 'bad', 'neutral'])].dropna(subset=['medical_diagnosis', 'suggested_follow_up'])
    df = df[df['class'].isin(['good', 'bad'])].dropna(subset=['medical_diagnosis', 'suggested_follow_up'])

    # texts = (df['reason_for_consultation'] + "<SEP>" + df['medical_diagnosis'] + df['prescription'] + "<SEP>" + df[
    #     'suggested_follow_up']).tolist()
    # texts = df['reason_for_consultation'].tolist()
    texts = df['summary'].tolist()
    labels = df['class'].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return texts, y, label_encoder


def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True)


class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=384, n_classes=2, hidden_dim=128, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.input_proj(x)  # [batch, hidden_dim]
        x = x.unsqueeze(1)  # [batch, seq_len=1, hidden_dim]
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)  # [batch, hidden_dim]
        return self.classifier(x)


def compute_weights(y, label_encoder):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights_tensor


# 5. Train and Evaluate
def train_model(df, batch_size=8, epochs=30, lr=1e-3):
    texts, y, label_encoder = prepare_data(df)
    X = get_embeddings(texts)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(input_dim=X.shape[1], n_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights_tensor = compute_weights(y_train, label_encoder).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    return model, label_encoder, SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), device


# 6. Inference
def predict_class(model, label_encoder, embedder, device, diag, follow_up):
    text = f"{diag} {follow_up}"
    embedding = embedder.encode([text], convert_to_numpy=True)
    tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]


if __name__ == '__main__':
    # Load data (JSON) using Dask for scalability
    data = get_data('medical_summaries_doctor_bias.json')
    # Use cached structured summaries if available, otherwise generate with LLM
    if not glob.glob('structured_summaries-*.csv'):
        df_structured = get_structured_summaries(data)
        print("Structured summaries saved to CSV.")
    else:
        df_structured = dd.read_csv('structured_summaries-*.csv')
    # Combine original data with structured results (column-wise)
    if isinstance(df_structured, dd.DataFrame) and isinstance(data, dd.DataFrame):
        # Ensure both dataframes have the same partitioning
        if df_structured.npartitions != data.npartitions:
            df_structured = df_structured.repartition(npartitions=data.npartitions)
        df = dd.concat([df_structured, data], axis=1)
    else:
        df = pd.concat([df_structured, data], axis=1)
    outcome_clusters_dict, df = cluster_summaries_and_outcomes(df)

    # Classify outcome clusters as good/neutral/bad and rank doctors
    future_outcome_classification = classify_medical_outcomes(outcome_clusters_dict)
    rank_doctors(df)
    model, le, embedder, device = train_model(df)
    prediction = predict_class(model, le, embedder, device, "Recovered from pneumonia", "return in 3 days")
    print("Predicted class:", prediction)
