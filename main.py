import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# --- Load Required Files ---
# Load the machine learning model, scaler for preprocessing, and the list of feature names used in the model
rf_model = joblib.load('rf_model.pkl')  # Random Forest model for CHD prediction
scaler = joblib.load('scaler.pkl')  # Scaler used to normalize numerical features
feature_names = joblib.load('feature_names.pkl')  # List of all features expected by the model

# --- Default Values for Optional Fields ---
# These values represent the averages or most common values from the dataset.
DEFAULT_VALUES = {
    'ChestPainType': 'ASY',  # Asymptomatic chest pain is the most common
    'RestingBP': 132.54,  # Average resting blood pressure (mm Hg)
    'Cholesterol': 244.64,  # Average cholesterol level (mg/dL)
    'FastingBS': 0,  # Majority of patients have fasting blood sugar ≤ 120 mg/dL
    'RestingECG': 'Normal',  # Normal ECG results are most common
    'MaxHR': 136.81,  # Average maximum heart rate achieved
    'ExerciseAngina': 'N',  # Most patients do not experience angina during exercise
    'Oldpeak': 0.89,  # Average ST depression induced by exercise
    'ST_Slope': 'Up'  # Up-sloping ST segment is the most common
}

# --- Preprocessing and Prediction ---
def preprocess_and_predict(data):
    """
    Preprocess user input data and make predictions using the trained model.
    
    Args:
        data (dict): Dictionary of user input values.
        
    Returns:
        tuple: Risk probability, risk category, and health suggestions.
    """
    # Replace 'Unknown' values with dataset averages
    for key, value in data.items():
        if value == 'Unknown':
            data[key] = DEFAULT_VALUES[key]

    # Convert the data dictionary to a DataFrame
    user_df = pd.DataFrame([data])

    # Encode categorical fields to match the model's expected input
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['RestingECG'] = pd.Categorical(user_df['RestingECG'], categories=['Normal', 'ST', 'LVH'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])
    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

    # Add any missing columns that are expected by the model, setting them to 0
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder the DataFrame columns to match the model's training data
    user_df = user_df[feature_names]

    # Scale numerical fields to match the preprocessing used during training
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )

    # Use the model to predict the risk probability
    risk_probability = rf_model.predict_proba(user_df)[0][1]  # Probability of having CHD

    # Categorize the risk probability into a risk category
    risk_category = (
        "Low Risk" if risk_probability < 0.2 else
        "Slight Risk" if risk_probability < 0.4 else
        "Moderate Risk" if risk_probability < 0.6 else
        "High Risk" if risk_probability < 0.8 else
        "Extreme Risk"
    )

    # Generate health suggestions based on the input data
    suggestions = generate_health_suggestions(data)

    return risk_probability, risk_category, suggestions

# --- Health Suggestions Generation ---
def generate_health_suggestions(data):
    """
    Generate personalized health suggestions based on the user's input data.
    
    Args:
        data (dict): Dictionary of user input values.
        
    Returns:
        list: List of suggestions for improving health or maintaining good health.
    """
    suggestions = []

    # --- Cholesterol ---
    if data['Cholesterol'] != DEFAULT_VALUES['Cholesterol']:
        if data['Cholesterol'] >= 240:
            suggestions.append(
                "Your cholesterol is critically high. Immediate dietary changes, such as avoiding fried and processed foods, are essential. "
                "Incorporate high-fiber foods (e.g., oatmeal, beans, lentils) and heart-healthy fats like those found in nuts and avocados. "
                "Medication such as statins might be needed—consult your doctor promptly."
            )
        elif 200 <= data['Cholesterol'] < 240:
            suggestions.append(
                "Your cholesterol level is high. Adopt a diet rich in fruits, vegetables, and lean proteins while avoiding trans fats. "
                "Regular physical activity and weight management can help improve your cholesterol profile."
            )
    
    # --- Resting Blood Pressure ---
    if data['RestingBP'] != DEFAULT_VALUES['RestingBP']:
        if data['RestingBP'] >= 140:
            suggestions.append(
                "Your resting blood pressure is in the hypertensive range. Reduce your sodium intake (target: under 1,500 mg daily), "
                "increase consumption of potassium-rich foods (e.g., bananas, potatoes, spinach), and avoid excessive caffeine or alcohol."
            )
        elif 120 <= data['RestingBP'] < 140:
            suggestions.append(
                "Your blood pressure is elevated. Adopt a heart-healthy diet like the DASH diet, manage stress, and engage in regular physical activity. "
                "Consider regular blood pressure monitoring to ensure it doesn't increase further."
            )
    
    # --- Fasting Blood Sugar ---
    if data['FastingBS'] != DEFAULT_VALUES['FastingBS']:
        if data['FastingBS'] == 1:
            suggestions.append(
                "Elevated fasting blood sugar suggests a risk for diabetes. Focus on reducing carbohydrate intake, especially from sugary drinks and refined grains. "
                "Incorporate regular exercise and consider losing weight if you're overweight. Speak with a healthcare provider about blood sugar management."
            )
    
    # --- Max Heart Rate ---
    if data['MaxHR'] != DEFAULT_VALUES['MaxHR']:
        if data['MaxHR'] < 85:
            suggestions.append(
                "Your maximum heart rate during physical activity appears to be quite low. This may indicate deconditioning or potential heart issues. "
                "Begin with light exercises, such as walking or yoga, and gradually increase intensity. Consult a doctor if fatigue or chest pain occurs during activity."
            )
        elif 85 <= data['MaxHR'] < 140:
            suggestions.append(
                "Your maximum heart rate is slightly below average. Consider increasing aerobic exercise to improve cardiovascular endurance. "
                "Aim for activities like cycling, brisk walking, or swimming at least 3–5 times per week."
            )
    
    # --- Exercise-Induced Angina ---
    if data['ExerciseAngina'] != DEFAULT_VALUES['ExerciseAngina']:
        if data['ExerciseAngina'] == 'Y':
            suggestions.append(
                "Exercise-induced angina (chest pain) is a warning sign of potential heart issues. Avoid strenuous activities until cleared by your doctor. "
                "Consult a healthcare provider for stress testing or imaging to evaluate your heart health."
            )
    
    # --- Oldpeak (ST depression) ---
    if data['Oldpeak'] != DEFAULT_VALUES['Oldpeak']:
        if data['Oldpeak'] > 2.0:
            suggestions.append(
                "ST depression (Oldpeak) greater than 2.0 may indicate significant heart issues such as ischemia. Immediate consultation with a cardiologist is recommended."
            )
        elif 1.0 <= data['Oldpeak'] <= 2.0:
            suggestions.append(
                "ST depression during exercise is slightly concerning. Focus on light-to-moderate physical activity under medical supervision. "
                "A thorough evaluation of your cardiovascular health may be necessary."
            )

    # Add general heart health tips if no specific suggestions were generated
    if not suggestions:
        suggestions.append(
            "Continue practicing heart-healthy habits such as:"
        )
        suggestions.extend([
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
            "Exercise regularly (at least 150 minutes of moderate activity per week).",
            "Avoid smoking and limit alcohol consumption.",
            "Manage stress effectively through mindfulness or relaxation techniques."
        ])

    return suggestions



def validate_input():
    """
    Validate the user inputs from the GUI before processing.
    
    - Ensures mandatory fields are filled.
    - Checks that optional fields have valid values if provided.
    - Substitutes 'Unknown' for optional fields left blank.
    
    Returns:
        dict: A dictionary containing the validated user inputs.
    """
    try:
        data = {}

        # --- Mandatory Input: Age ---
        if not age_entry.get().strip():  # Ensure the Age field is not empty
            raise ValueError("Age is required.")
        data['Age'] = float(age_entry.get())  # Convert input to float for processing
        if not (0 <= data['Age'] <= 120):  # Check if age is within a valid range
            raise ValueError("Age must be between 0 and 120.")

        # --- Mandatory Input: Sex ---
        if not sex_var.get():  # Ensure the Sex field is selected
            raise ValueError("Sex is required.")
        data['Sex'] = sex_var.get().upper()  # Convert input to uppercase for consistency

        # --- Optional Inputs ---
        # Chest Pain Type
        data['ChestPainType'] = chest_pain_var.get().upper() if chest_pain_var.get() else 'Unknown'
        
        # Resting Blood Pressure
        data['RestingBP'] = float(bp_entry.get()) if bp_entry.get().strip() else 'Unknown'
        if data['RestingBP'] != 'Unknown' and not (80 <= data['RestingBP'] <= 200):
            raise ValueError("Resting BP must be between 80 and 200 mm Hg.")

        # Cholesterol
        data['Cholesterol'] = float(cholesterol_entry.get()) if cholesterol_entry.get().strip() else 'Unknown'
        if data['Cholesterol'] != 'Unknown' and not (100 <= data['Cholesterol'] <= 400):
            raise ValueError("Cholesterol must be between 100 and 400 mg/dL.")

        # Fasting Blood Sugar
        data['FastingBS'] = int(fbs_var.get()) if fbs_var.get() else 'Unknown'

        # Resting ECG
        data['RestingECG'] = ecg_var.get().capitalize() if ecg_var.get() else 'Unknown'

        # Maximum Heart Rate
        data['MaxHR'] = float(maxhr_entry.get()) if maxhr_entry.get().strip() else 'Unknown'
        if data['MaxHR'] != 'Unknown' and not (60 <= data['MaxHR'] <= 220):
            raise ValueError("Max Heart Rate must be between 60 and 220 bpm.")

        # Exercise-Induced Angina
        data['ExerciseAngina'] = angina_var.get().upper() if angina_var.get() else 'Unknown'

        # Oldpeak (ST Depression)
        data['Oldpeak'] = float(oldpeak_entry.get()) if oldpeak_entry.get().strip() else 'Unknown'
        if data['Oldpeak'] != 'Unknown' and not (0.0 <= data['Oldpeak'] <= 6.0):
            raise ValueError("Oldpeak must be between 0.0 and 6.0.")

        # ST Slope
        data['ST_Slope'] = st_slope_var.get().capitalize() if st_slope_var.get() else 'Unknown'

        return data

    except ValueError as e:
        # Display an error message if validation fails
        messagebox.showerror("Input Error", str(e))
        return None


def display_risk(probability, category, suggestions):
    """
    Display the CHD risk probability, category, and health suggestions in a new window.
    
    Args:
        probability (float): The predicted probability of CHD.
        category (str): The risk category based on the probability.
        suggestions (list): List of personalized health suggestions.
    """
    # Define background colors for each risk category
    category_colors = {
        "Low Risk": "#d4edda",  # Light green
        "Slight Risk": "#fff3cd",  # Light yellow
        "Moderate Risk": "#ffeeba",  # Light orange
        "High Risk": "#f8d7da",  # Light red
        "Extreme Risk": "#f5c6cb"  # Bright red
    }

    # Create a new window to display results
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Result")
    result_window.geometry("500x600")
    result_window.configure(bg=category_colors.get(category, "white"))  # Set background color based on risk category

    # Display CHD risk probability
    tk.Label(
        result_window,
        text=f"CHD Risk Probability: {probability * 100:.2f}%",
        font=("Helvetica", 16, "bold"),
        bg=category_colors.get(category, "white")
    ).pack(pady=10)

    # Display risk category
    tk.Label(
        result_window,
        text=f"Risk Category: {category}",
        font=("Helvetica", 20, "bold"),
        bg=category_colors.get(category, "white"),
        fg="black"
    ).pack(pady=10)

    # Display health suggestions
    if suggestions:
        tk.Label(
            result_window,
            text="Suggestions to Improve Your Health:",
            font=("Helvetica", 14, "bold"),
            bg=category_colors.get(category, "white"),
            fg="black"
        ).pack(pady=10)
        for suggestion in suggestions:
            tk.Label(
                result_window,
                text=f"• {suggestion}",
                font=("Arial", 12),
                bg=category_colors.get(category, "white"),
                wraplength=450,  # Wrap text to fit within the window
                justify="left"
            ).pack(anchor="w", padx=20)
    else:
        tk.Label(
            result_window,
            text="No specific suggestions at this time.",
            font=("Arial", 12),
            bg=category_colors.get(category, "white")
        ).pack(pady=10)

    # Add a close button
    tk.Button(
        result_window,
        text="Close",
        command=result_window.destroy,
        font=("Arial", 14)
    ).pack(pady=20)


def on_submit():
    """
    Handle the submit button click.
    
    - Validates user input.
    - Preprocesses the input and makes a prediction.
    - Displays the prediction results and suggestions.
    """
    if show_informative_dialog():
        user_data = validate_input()  # Validate the user inputs
        if user_data is not None:  # Proceed if inputs are valid
            try:
                # Preprocess data, predict risk, and generate suggestions
                probability, category, suggestions = preprocess_and_predict(user_data)
                display_risk(probability, category, suggestions)  # Display results
            except Exception as e:
                # Show an error message if the prediction fails
                messagebox.showerror("Processing Error", f"An error occurred during prediction: {e}")


def show_informative_dialog():
    """
    Show a dialog box explaining how missing fields are handled.
    
    Returns:
        bool: True if the user chooses to proceed, False otherwise.
    """
    if not all([bp_entry.get(), cholesterol_entry.get(), maxhr_entry.get(), oldpeak_entry.get()]):
        # Show a warning about incomplete data and ask for confirmation to proceed
        response = messagebox.askyesno(
            "Incomplete Data",
            "Some optional fields are missing.\n"
            "For the most accurate prediction, please fill in as many fields as possible.\n\n"
            "Fields set to 'Unknown' or left blank will use average values from the training dataset for predictions.\n\n"
            "Do you want to proceed?"
        )
        return response
    return True


# --- GUI Setup ---
# Main application window setup
root = tk.Tk()
root.title("CHD Risk Prediction")  # Title of the application
root.geometry("500x900")  # Adjusted to a smaller window height
root.configure(bg="#f5f5f5")  # Background color

# Apply styles to GUI elements
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), background="#f5f5f5")
style.configure("TEntry", font=("Arial", 12))
style.configure("TButton", font=("Arial", 14), padding=5)
style.configure("TCombobox", font=("Arial", 12))

# Create a Canvas widget for the scrollbar
canvas = tk.Canvas(root, bg="#f5f5f5", highlightthickness=0)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas to work with the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Frame for form inputs inside the canvas
form_frame = ttk.Frame(canvas, padding=(10, 0, 0, 0))  # Add 10px padding on the left
canvas.create_window((0, 0), window=form_frame, anchor="nw", width=480)


# Header for the application
header = tk.Label(form_frame, text="CHD Risk Prediction", font=("Helvetica", 16, "bold"), bg="#0078d7", fg="white", pady=10)
header.pack(fill="x")

# Input field creation helper function
def add_input_field(label, description, var_type, options=None):
    """
    Create input fields for the GUI form.
    
    Args:
        label (str): Label for the input field.
        description (str): Description of the input field.
        var_type (type): Data type of the input field.
        options (list, optional): List of options for dropdown menus.
        
    Returns:
        tk.StringVar or ttk.Entry: The input widget or variable for the field.
    """
    ttk.Label(form_frame, text=label).pack(anchor="w", pady=2)
    tk.Label(form_frame, text=description, font=("Arial", 10), fg="gray", wraplength=450).pack(anchor="w", pady=2)
    if options:
        var = tk.StringVar()
        field = ttk.Combobox(form_frame, textvariable=var, values=options, state="readonly")
        field.pack(fill="x", pady=2)
        field.current(0)  # Set default selection to the first option
        return var
    else:
        field = ttk.Entry(form_frame)
        field.pack(fill="x", pady=2)
        return field
    
def on_mousewheel(event):
    canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)


# Input fields for the form
age_entry = add_input_field("Age (Years)", "Age of the patient (0–120 years).", int)
sex_var = add_input_field("Sex", "Sex of the patient [M: Male, F: Female].", str, ["M", "F"])
chest_pain_var = add_input_field(
    "Chest Pain Type", 
    "Type of chest pain:\nTA: Typical Angina\nATA: Atypical Angina\nNAP: Non-Anginal Pain\nASY: Asymptomatic.", 
    str, ["TA", "ATA", "NAP", "ASY", "Unknown"]
)
bp_entry = add_input_field("Resting BP (mm Hg)", "Resting blood pressure in mm Hg (80–200).", int)
cholesterol_entry = add_input_field("Cholesterol (mg/dL)", "Serum cholesterol in mg/dL (100–400).", int)
fbs_var = add_input_field("Fasting BS (0/1)", "Fasting blood sugar [1: >120 mg/dL, 0: otherwise, Unknown].", int, ["0", "1", "Unknown"])
ecg_var = add_input_field(
    "Resting ECG", 
    "Resting electrocardiogram results:\nNormal: Normal\nST: ST-T wave abnormality\nLVH: Left ventricular hypertrophy.", 
    str, ["Normal", "ST", "LVH", "Unknown"]
)
maxhr_entry = add_input_field("Max Heart Rate", "Maximum heart rate achieved (60–220).", int)
angina_var = add_input_field("Exercise Angina", "Exercise-induced angina [Y: Yes, N: No, Unknown].", str, ["Y", "N", "Unknown"])
oldpeak_entry = add_input_field("Oldpeak", "ST depression induced by exercise (0.0–6.0).", float)
st_slope_var = add_input_field(
    "ST Slope", 
    "Slope of the peak exercise ST segment:\nUp: Upsloping\nFlat: Flat\nDown: Downsloping.", 
    str, ["Up", "Flat", "Down", "Unknown"]
)

# Submit button
submit_button = ttk.Button(form_frame, text="Submit", command=on_submit)
submit_button.pack(pady=20)

# Footer
footer = tk.Label(form_frame, text="Powered by Machine Learning", font=("Arial", 10), bg="#f5f5f5", fg="gray")
footer.pack(side="bottom", pady=10)

# Start the GUI event loop
root.mainloop()
