import torch
from transformers import CLIPProcessor, CLIPModel, BartForConditionalGeneration, BartTokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import cv2
import os
import pandas as pd

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("./clip_model")
clip_processor = CLIPProcessor.from_pretrained("./clip_model")

tokenizer = PegasusTokenizer.from_pretrained("./pegasus_xsum")
model = PegasusForConditionalGeneration.from_pretrained("./pegasus_xsum")

# Function to extract frames from a video
def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_rate == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames

# Function to preprocess frames for CLIP
def preprocess_frames(frames):
    images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    return inputs

# Function to extract keywords using CLIP
def extract_keywords(frames):
    keywords = ["dog", "cat", "nature", "outdoor", "city", "action"]  # Add relevant keywords
    text_inputs = clip_processor(text=keywords, return_tensors="pt", padding=True)
    images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    image_inputs = clip_processor(images=images, return_tensors="pt", padding=True)

    # Compute similarity between image and text
    outputs = clip_model(pixel_values=image_inputs["pixel_values"], input_ids=text_inputs["input_ids"])
    logits_per_image = outputs.logits_per_image  # Image-to-text scores
    best_keywords = [keywords[idx] for idx in logits_per_image.argmax(dim=1).tolist()]
    return list(set(best_keywords))  # Deduplicate keywords

def process_annotations(video_annotations):
    # Split the importance scores (column 2) into lists of integers
    video_annotations.loc[:, 2] = video_annotations[2].apply(
        lambda x: list(map(int, x.split(','))) if isinstance(x, str) else []
    )

    # Debug: Print the first row to check processed annotations
    print("Processed importance scores (first row):", video_annotations.iloc[0, 2][:10])  # First 10 scores
    return video_annotations


# Function to generate textual summaries using BART
def generate_summary(keywords):
    top_keywords = keywords[:3]  # Limit to 3 most relevant keywords
    prompt = f"This video highlights: {', '.join(top_keywords)}. Summarize the main points."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,  # Keep the summary concise
        min_length=30,
        num_beams=3     # Beam search for better output
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Save keyframes to a directory
def save_keyframes(frames, output_dir="keyframes"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"keyframe_{idx + 1}.jpg"), frame)

# Evaluate summary with TVSum annotations
def evaluate_summary(generated_keyframes, video_annotations):
    # Ensure annotations are processed
    video_annotations = process_annotations(video_annotations)

    # Extract high-importance frames (example threshold: >4)
    high_importance_frames = [
        idx for idx, score in enumerate(video_annotations.iloc[0, 2]) if score > 4
    ]

    print("High-importance frames:", high_importance_frames[:10])  # Debug: first 10
    print("Generated keyframes:", generated_keyframes[:10])  # Debug: first 10

    # Evaluate keyframes
    matched = len(set(generated_keyframes) & set(high_importance_frames))
    precision = matched / len(generated_keyframes) if generated_keyframes else 0
    recall = matched / len(high_importance_frames) if high_importance_frames else 0

    return precision, recall

def get_generated_keyframes(frames, step=None):
    # Dynamically set step to get a similar number of keyframes as high-importance frames
    if step is None:
        step = max(1, len(frames) // 10)  # Aim for ~10 keyframes
    return [i for i in range(0, len(frames), step)]

def summarize_video(video_path, annotations=None):
    print("Extracting frames...")
    frames = extract_frames(video_path)

    print("Processing frames with CLIP...")
    inputs = preprocess_frames(frames)

    # Add dummy text input (required for CLIP text processing)
    dummy_texts = [""] * inputs["pixel_values"].shape[0]
    inputs.update(clip_processor(text=dummy_texts, return_tensors="pt", padding=True))

    keywords = extract_keywords(frames)

    print("Generating summaries with BART...")
    summary = generate_summary(keywords)

    # Select generated keyframes
    generated_keyframes = get_generated_keyframes(frames)

    # If annotations are provided, evaluate the summary
    if annotations is not None:
        precision, recall = evaluate_summary(generated_keyframes=generated_keyframes, video_annotations=annotations)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

    return frames, summary


# Example usage
if __name__ == "__main__":
    # Load TVSum annotations
    annotations = pd.read_csv("ydata-tvsum50-anno.tsv", sep="\t", header=None)

    # Define video path and ID
    video_id = "AwmHb44_ouw"  # Use the correct video ID
    video_path = "TVSum/video/AwmHb44_ouw.mp4"  # Update the file path accordingly

    # Filter annotations for the video
    video_annotations = annotations[annotations[0] == video_id]

    # Summarize video and evaluate
    frames, summary = summarize_video(video_path, annotations=video_annotations)
    print("Summary:", summary)

    # Save keyframes for inspection
    save_keyframes(frames)



    # Dog Video Example (for later testing)
    # To test with the dog video, replace the following lines:
    # video_path = "dog_short_vid.mp4"
    # frames, summary = summarize_video(video_path)
    # print("Summary:", summary)
    # save_keyframes(frames)
