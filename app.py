import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# CSV-Daten laden mit Komma als Dezimaltrennzeichen
df = pd.read_csv("rezeptdaten.csv", encoding="utf-8", sep=";", decimal=",")

# Spaltennamen bereinigen
df.columns = df.columns.str.strip()

# Zielgrößen definieren
targets = [
    "Glanz 20", "Glanz 60", "Glanz 85",
    "Viskosität lowshear", "Viskosität midshear", "Brookfield",
    "Kosten Gesamt kg"
]
targets = [t.strip() for t in targets]

# Existierende Zielspalten prüfen
existing_targets = [col for col in targets if col in df.columns]
if len(existing_targets) < len(targets):
    fehlende = set(targets) - set(existing_targets)
    st.warning(f"Diese Zielspalten fehlen im Datensatz und werden ignoriert: {fehlende}")

# Zielspalten in numerisch umwandeln (ungültige Werte zu NaN)
for col in existing_targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eingabe- und Ausgabedaten trennen
X = df.drop(columns=existing_targets)
y = df[existing_targets]

# Kategorische und numerische Variablen erkennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding der Eingabedaten
X_encoded = pd.get_dummies(X)

# NaNs in Features (X_encoded) mit 0 füllen
X_encoded = X_encoded.fillna(0)

# Nur Zeilen mit vollständigen Zielwerten behalten
mask = y.notna().all(axis=1)
X_encoded_clean = X_encoded.loc[mask]
y_clean = y.loc[mask]

# Debug-Ausgaben vor Modelltraining
st.write(f"Shape X_encoded_clean: {X_encoded_clean.shape}")
st.write(f"Shape y_clean: {y_clean.shape}")
st.write("NaN-Werte in y_clean:")
st.write(y_clean.isna().sum())
st.write("y_clean Datentypen:")
st.write(y_clean.dtypes)
st.write("y_clean Beispiele:")
st.write(y_clean.head())

# Modell trainieren mit korrektem Datentyp
if y_clean.shape[1] == 1:
    y_array = y_clean.values.ravel()
    modell = RandomForestRegressor(n_estimators=150, random_state=42)
    modell.fit(X_encoded_clean, y_array)
else:
    y_array = y_clean.values.astype(float)
    modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
    modell.fit(X_encoded_clean, y_array)

# Streamlit UI
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# Eingabeformular in Sidebar
user_input = {}
st.sidebar.header("🔧 Eingabewerte anpassen")

for col in numerisch:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende One-Hot-Spalten ergänzen
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# Vorhersage
if y_clean.shape[1] == 1:
    prediction = modell.predict(input_encoded)[0]
    st.subheader("🔮 Vorhergesagte Eigenschaft")
    st.metric(label=existing_targets[0], value=round(prediction, 2))
else:
    prediction = modell.predict(input_encoded)[0]
    st.subheader("🔮 Vorhergesagte Eigenschaften")
    for i, ziel in enumerate(existing_targets):
        st.metric(label=ziel, value=round(prediction[i], 2))
