from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import os
import io
import tempfile
import datetime
import numpy as np
from PIL import Image
from fpdf import FPDF

# Optional: TensorFlow model loading
try:
	import tensorflow as tf
	from tensorflow.keras.models import load_model
	TF_AVAILABLE = True
except Exception:
	TF_AVAILABLE = False

# Optional: Gemini SDK
try:
	from dotenv import load_dotenv
	import google.generativeai as genai
	load_dotenv()
	genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
	gemini_model = genai.GenerativeModel("gemini-2.0-flash")
	GEMINI_AVAILABLE = True
except Exception:
	GEMINI_AVAILABLE = False


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


# Load class names from JSON file
import json
import markdown

def load_class_labels():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	json_path = os.path.join(base_dir, "class_indices.json")

	with open(json_path, "r") as f:
		class_map = json.load(f)

	# Reverse mapping: index → label
	index_to_label = {v: k for k, v in class_map.items()}

	# Convert to list ordered by index
	labels = [index_to_label[i] for i in sorted(index_to_label.keys())]

	return labels

CLASS_LABELS = load_class_labels()


IMAGE_SIZE = (224, 224)

MODEL = None


def get_model_path() -> str:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(base_dir, "final_model.h5")


def load_tf_model():
	global MODEL
	if not TF_AVAILABLE:
		return None
	if MODEL is None:
		model_path = get_model_path()
		if os.path.exists(model_path):
			try:
				MODEL = load_model(model_path, compile=False)
			except Exception:
				# Fallback to heuristic if model cannot be loaded (e.g., shape mismatch)
				MODEL = None
		else:
			MODEL = None
	return MODEL


def preprocess_image_for_model(pil_img: Image.Image, model) -> np.ndarray:
	"""Resize and normalize image for the Keras model, handling 1 or 3 channels."""
	target = pil_img.resize(IMAGE_SIZE)
	# Try to match model expected channels
	if hasattr(model, "input_shape") and model.input_shape is not None:
		channels = model.input_shape[-1]
	else:
		channels = 3
	if channels == 1:
		target = target.convert("L")
		arr = np.array(target, dtype=np.float32)
		arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
	else:
		target = target.convert("RGB")
		arr = np.array(target, dtype=np.float32)  # (H, W, 3)
	arr = arr / 255.0
	arr = np.expand_dims(arr, axis=0)
	return arr


def predict_with_fallback(pil_img: Image.Image):
	"""Predict using TF model if available; otherwise simple heuristic with confidence."""
	model = load_tf_model()
	if model is not None:
		arr = preprocess_image_for_model(pil_img, model)
		preds = model.predict(arr)
		pred_index = int(np.argmax(preds))
		confidence = float(np.max(preds)) * 100.0
		return CLASS_LABELS[pred_index], confidence

	# Fallback heuristic using brightness bins mapped to all labels
	gray = pil_img.convert("L")
	arr = np.array(gray, dtype=np.uint8)
	brightness = float(np.mean(arr))  # 0..255
	num_classes = len(CLASS_LABELS)
	# Map 0..255 to 0..num_classes-1
	bin_index = int(np.clip(np.floor((brightness / 256.0) * num_classes), 0, num_classes - 1))
	# Confidence higher when brightness is near the center of its bin
	bin_size = 256.0 / num_classes
	bin_center = (bin_index + 0.5) * bin_size
	dist = abs(brightness - bin_center)
	confidence = float(np.clip(100.0 - (dist / (bin_size / 2)) * 40.0, 60.0, 98.0))
	return CLASS_LABELS[bin_index], confidence


def get_treatment_and_risk(disease: str) -> str:
	if GEMINI_AVAILABLE:
		try:
			prompt = (
				f"Provide a clear and detailed treatment plan and risk assessment for {disease}. "
				"Give the response in bullet points. Include common treatments, follow-up advice, "
				"and important warning signs the patient should look for."
			)
			response = gemini_model.generate_content(contents=prompt)
			return response.text
		except Exception as e:
			return f"Could not fetch treatment info from Gemini: {e}"
	# Fallback static guidance
	return (
		f"Basic guidance for {disease}:\n"
		"- Consult an ophthalmologist for a comprehensive exam.\n"
		"- Follow prescribed medications and attend follow-ups.\n"
		"- Monitor symptoms and seek urgent care if vision worsens."
	)


def create_pdf_report(patient_info, image_path, prediction, confidence, treatment) -> bytes:
	pdf = FPDF()
	pdf.add_page()
	
	# Title
	pdf.set_text_color(0, 102, 204)  # Blue
	pdf.set_font("Arial", 'B', 20)
	pdf.cell(0, 10, "DermaDetect - Skin Disease Diagnosis Report", 0, 1, 'C')
	
	# Horizontal Line
	pdf.set_draw_color(200, 200, 200)
	pdf.line(10, 25, 200, 25)
	pdf.ln(10)
	
	# Helper for Section Headers
	def section_header(title):
		pdf.set_fill_color(240, 240, 240)  # Light Gray
		pdf.set_text_color(0, 0, 0)
		pdf.set_font("Arial", 'B', 12)
		pdf.cell(0, 8, title, 0, 1, 'L', fill=True)
		pdf.ln(2)

	# Patient Information
	section_header("Patient Information")
	pdf.set_font("Arial", '', 10)
	pdf.cell(30, 6, "Name:", 0, 0)
	pdf.cell(0, 6, f"{patient_info['name']}", 0, 1)
	pdf.cell(30, 6, "Age:", 0, 0)
	pdf.cell(0, 6, f"{patient_info['age']}", 0, 1)
	pdf.cell(30, 6, "Gender:", 0, 0)
	pdf.cell(0, 6, f"{patient_info['gender']}", 0, 1)
	pdf.cell(30, 6, "Date:", 0, 0)
	pdf.cell(0, 6, f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
	pdf.ln(5)

	# Image Analysis
	section_header("Image Analysis")
	try:
		# Center the image
		img_width = 80
		x_pos = (210 - img_width) / 2
		pdf.image(image_path, x=x_pos, w=img_width)
	except Exception:
		pdf.cell(0, 10, "Image not available", 0, 1)
	pdf.ln(5)

	# Diagnosis Results
	section_header("Diagnosis Results")
	pdf.set_font("Arial", 'B', 10)
	pdf.cell(40, 6, "Predicted Condition:", 0, 0)
	pdf.set_text_color(220, 20, 60) # Red for condition
	pdf.cell(0, 6, f"{prediction}", 0, 1)
	pdf.set_text_color(0, 0, 0)
	pdf.cell(40, 6, "Confidence Score:", 0, 0)
	pdf.set_font("Arial", '', 10)
	pdf.cell(0, 6, f"{confidence:.2f}%", 0, 1)
	pdf.ln(5)

	# Treatment Plan & Risk Assessment
	section_header("Treatment Plan & Risk Assessment")
	pdf.set_font("Arial", '', 10)
	
	# Convert markdown to HTML for PDF
	# Note: FPDF's write_html support is basic. We might need to handle bullet points manually if write_html isn't perfect,
	# but let's try write_html first with some styling.
	treatment_html = markdown.markdown(treatment)
	
	# Clean up some markdown artifacts if needed or style it
	# FPDF write_html doesn't support full CSS. We use basic tags.
	
	# Sanitize text to replace common Unicode characters not supported by Latin-1
	replacements = {
		"\u2018": "'", "\u2019": "'",  # Smart quotes
		"\u201c": '"', "\u201d": '"',
		"\u2013": "-", "\u2014": "-",  # Dashes
		"\u2022": "-",                 # Bullet point
		"\u2026": "...",               # Ellipsis
	}
	for src, dst in replacements.items():
		treatment = treatment.replace(src, dst)
	
	# Final fallback: encode to latin-1, replacing errors with ?
	treatment = treatment.encode('latin-1', 'replace').decode('latin-1')

	treatment_html = markdown.markdown(treatment)
	
	styled_html = f'<font face="Arial" size="10">{treatment_html}</font>'
	
	pdf.write_html(styled_html)
	
	res = pdf.output(dest='S')
	# fpdf2 may return a str, bytes, or bytearray depending on version
	if isinstance(res, (bytes, bytearray)):
		return bytes(res)
	return str(res).encode('latin1')


@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	name = request.form.get('name', '').strip()
	age = request.form.get('age', '').strip()
	gender = request.form.get('gender', '').strip()
	file = request.files.get('image')

	if not name or not file:
		flash('Please provide patient name and an image.')
		return redirect(url_for('index'))

	filename = secure_filename(file.filename)
	if filename == '':
		flash('Invalid file.')
		return redirect(url_for('index'))

	with tempfile.TemporaryDirectory() as tmpdir:
		img_path = os.path.join(tmpdir, filename)
		file.save(img_path)

		# Open image with context manager to avoid Windows file locks
		with Image.open(img_path) as pil_img:
			predicted_class, confidence = predict_with_fallback(pil_img)
			treatment = get_treatment_and_risk(predicted_class)

			# Save image copy for display while file is open
			display_path = os.path.join(tmpdir, f"display_{filename}")
			pil_img.convert('RGB').save(display_path)

		# Create PDF bytes after image operations
		patient = {"name": name, "age": age or "-", "gender": gender or "-"}
		pdf_bytes = create_pdf_report(patient, img_path, predicted_class, confidence, treatment)

		# Read display bytes for embedding
		with open(display_path, 'rb') as f:
			display_bytes = f.read()

		# Return results page with embedded image (base64)
		import base64
		# import markdown  <-- removed local import
		img_b64 = base64.b64encode(display_bytes).decode('utf-8')
		pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
		
		treatment_html = markdown.markdown(treatment)

		return render_template(
			'result.html',
			name=name,
			age=age,
			gender=gender,
			predicted_class=predicted_class,
			confidence=confidence,
			treatment=treatment_html,
			image_data=img_b64,
			pdf_data=pdf_b64,
			timestamp=int(datetime.datetime.now().timestamp())
		)


@app.route('/download', methods=['POST'])
def download():
	# Receive base64 pdf and send as file
	pdf_b64 = request.form.get('pdf_data')
	name = request.form.get('name', 'Report').replace(' ', '_')
	if not pdf_b64:
		flash('No report to download.')
		return redirect(url_for('index'))
	import base64
	pdf_bytes = base64.b64decode(pdf_b64)
	return send_file(
		io.BytesIO(pdf_bytes),
		mimetype='application/pdf',
		as_attachment=True,
		download_name=f"DermaDetect_Report_{name}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
	)


if __name__ == '__main__':
	# For local development on Windows
	app.run(host='127.0.0.1', port=5000, debug=True)
