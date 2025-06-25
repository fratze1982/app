import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# CSV-Daten laden
df = pd.read_csv("rezeptdaten.csv", encoding="utf-8", sep=";")

# Spaltennamen säubern (Leerzeichen am Anfang/Ende entfernen)
df.columns = df.columns.str.strip()

# Zielgrößen definieren
targets = [
    "Glanz 20", "Glanz 60", "Glanz 85",
    "Viskosität lowshear", "Viskosität midshear", "Brookfield",
    "Kosten Gesamt kg"
]

# Säubere auch die Zielnamen (falls hier Leerzeichen drin sind)
targets = [t.strip() for t in targets]

# Überprüfen, welche Zielspalten tatsächlich im DataFrame sind
existing_targets = [col for col in targets if col in df.columns]
if len(existing_targets) < len(targets):
    fehlende = set(targets) - set(existing_targets)
    st.warning(f"Diese Zielspalten fehlen im Datensatz und werden ignoriert: {fehlende}")

# Fehlende Werte in Zielspalten entfernen
df_clean = df.dropna(subset=existing_targets)

# Eingabe- und Ausgabedaten trennen
X = df_clean.drop(columns=existing_targets)
y = df_clean[existing_targets].astype(float)  # Wichtig: in float umwandeln

# Kategorische und numerische Variablen erkennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding der Eingabedaten
X_encoded = pd.get_dummies(X)

# Debug-Ausgaben vor Training (kannst du später auskommentieren)
st.write("Shape X_encoded:", X_encoded.shape)
st.write("Shape y:", y.shape)
st.write("Zieldaten Beispiel:")
st.write(y.head())

# Modell trainieren
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded, y)

# Streamlit UI
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# Eingabeformular in der Sidebar
user_input = {}
st.sidebar.header("🔧 Eingabewerte anpassen")

# Numerische Eingaben als Slider
for col in numerisch:
    min_val = float(df_clean[col].min())
    max_val = float(df_clean[col].max())
    mean_val = float(df_clean[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

# Kategorische Eingaben als Dropdown
for col in kategorisch:
    options = sorted(df_clean[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

# Eingabe in DataFrame und One-Hot-Encoding
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende One-Hot-Spalten ergänzen
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Spalten in korrekter Reihenfolge sortieren
input_encoded = input_encoded[X_encoded.columns]

# Vorhersage berechnen
prediction = modell.predict(input_encoded)[0]

# Ergebnisse anzeigen
st.subheader("🔮 Vorhergesagte Eigenschaften")
for i, ziel in enumerate(existing_targets):
    st.metric(label=ziel, value=round(prediction[i], 2))


