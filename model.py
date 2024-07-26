import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
dataset = pd.read_csv('medtronic_data.csv')

dataset = dataset.drop(columns=['patient_id', 'cadence_0_weeks', 'cadence_8_weeks', 'cadence_24_weeks', 'cadence_52_weeks', 'step_length_0_weeks', 'step_length_8_weeks', 
                                'step_length_24_weeks', 'step_length_52_weeks', 'step_width_0_weeks', 'step_width_8_weeks', 'step_width_24_weeks', 'step_width_52_weeks'])
dataset.to_csv('anon_medtronic_data.csv', index=False)
print(dataset.columns)

# Define the parameters to be normalized
parameters = ['knee_ext_1_wk', 'knee_ext_2_wk', 'knee_ext_4_wk', 'knee_ext_8_wk', 'knee_ext_12_wk', 'knee_ext_24_wk',
              'knee_flx_1_wk', 'knee_flx_2_wk', 'knee_flx_4_wk', 'knee_flx_8_wk', 'knee_flx_12_wk', 'knee_flx_24_wk',
              'kin_180_acl_recon_4_wk', 'kin_180_acl_recon_8_wk', 'kin_180_acl_recon_12_wk', 'kin_180_acl_recon_24_wk',
              'kin_60_acl_recon_4_wk', 'kin_60_acl_recon_8_wk', 'kin_60_acl_recon_12_wk', 'kin_60_acl_recon_24_wk']


missing_columns = [col for col in parameters if col not in dataset.columns]
if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
else:
   
    scaler = MinMaxScaler()
    normalized_parameters = scaler.fit_transform(dataset[parameters])

    joblib.dump(scaler, 'scaler.pkl')

   
    normalized_dataset = pd.DataFrame(normalized_parameters, columns=parameters)

   
    weights = {
        'knee_ext_1_wk': 0.125, 'knee_ext_2_wk': 0.06328125, 'knee_ext_4_wk': 0.03203125, 'knee_ext_8_wk': 0.01640625, 'knee_ext_12_wk': 0.00859375, 'knee_ext_24_wk': 0.0046875,
        'knee_flx_1_wk': 0.125, 'knee_flx_2_wk': 0.06328125, 'knee_flx_4_wk': 0.03203125, 'knee_flx_8_wk': 0.01640625, 'knee_flx_12_wk': 0.00859375, 'knee_flx_24_wk': 0.0046875,
        'kin_180_acl_recon_4_wk': 0.0625, 'kin_180_acl_recon_8_wk': 0.0625, 'kin_180_acl_recon_12_wk': 0.0625, 'kin_180_acl_recon_24_wk': 0.0625,
        'kin_60_acl_recon_4_wk': 0.0625, 'kin_60_acl_recon_8_wk': 0.0625, 'kin_60_acl_recon_12_wk': 0.0625, 'kin_60_acl_recon_24_wk': 0.0625
    }


    normalized_dataset['Composite_Score'] = sum(normalized_dataset[col] * weight for col, weight in weights.items())
    dataset['Composite_Score'] = normalized_dataset['Composite_Score']

    print(dataset['knee_ext_1_wk'][9967])
    print(dataset['Composite_Score'][9967])
    # Define features and target
    X = dataset.drop(columns=['Composite_Score'])
    y = dataset['Composite_Score']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Regression output
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")

    # Save the model
    model.save('knee_recovery_model.h5')


    # Plot training & validation loss and MAE
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()

    # Evaluate residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred.flatten()

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual Composite Score')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.show()

    # Additional performance metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")