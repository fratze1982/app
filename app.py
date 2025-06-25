import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# CSV-Daten laden
df = pd.read_csv("rezeptdaten.csv", encoding="utf-8", sep=";")

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

# Zielspalten in numerisch umwandeln, nicht-konvertierbare Werte werden NaN
for col in existing_targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Zeilen mit NaN in Zielspalten entfernen
df_clean = df.dropna(subset=existing_targets)

# Eingabe- und Ausgabedaten trennen
X = df_clean.drop(columns=existing_targets)
y = df_clean[existing_targets]

# Kategorische und numerische Variablen erkennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding der Eingabedaten
X_encoded = pd.get_dummies(X)

# Debug-Ausgaben (optional)
st.write("Shape X_encoded:", X_encoded.shape)
st.write("Shape y:", y.shape)
st.write("Datentypen y:")
st.write(y.dtypes)
st.write("NaN-Werte in y:")
st.write(y.isna().sum())

# Modell dynamisch wählen
if len(existing_targets) == 1:
    # Einzelziel: y als 1D-Array, RandomForestRegressor direkt nutzen
    y_single = y.values.ravel()
    modell = RandomForestRegressor(n_estimators=150, random_state=42)
    modell.fit(X_encoded, y_single)
else:
    # Mehrziel: MultiOutputRegressor nutzen
    modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
    modell.fit(X_encoded, y)

# Streamlit UI
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# Eingabeformular in Sidebar
user_input = {}
st.sidebar.header("🔧 Eingabewerte anpassen")

for col in numerisch:
    min_val = float(df_clean[col].min())
    max_val = float(df_clean[col].max())
    mean_val = float(df_clean[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    options = sorted(df_clean[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende One-Hot-Spalten ergänzen
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# Vorhersage
if len(existing_targets) == 1:
    prediction = modell.predict(input_encoded)[0]
    st.subheader("🔮 Vorhergesagte Eigenschaft")
    st.metric(label=existing_targets[0], value=round(prediction, 2))
else:
    prediction = modell.predict(input_encoded)[0]
    st.subheader("🔮 Vorhergesagte Eigenschaften")
    for i, ziel in enumerate(existing_targets):
        st.metric(label=ziel, value=round(prediction[i], 2))
