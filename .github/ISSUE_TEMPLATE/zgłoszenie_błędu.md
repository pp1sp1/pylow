---
name: Bug report
about: Zgłoś błąd kompilatora (ICE, błędna optymalizacja lub nieprawidłowy błąd)
title: '[Komponent] Krótki opis problemu'
labels: 'bug'
assignees: ''

---

## 🐛 Opis błędu
Jasny i zwięzły opis błędu kompilatora.
*(np. Kompilator kończy pracę awarią systemu, generuje nieprawidłowy kod wynikowy lub odrzuca poprawny kod źródłowy).*

---

## 💻 Informacje o środowisku
*Proszę podać szczegóły dotyczące środowiska, w którym uruchamiany jest kompilator.*

- **Wersja kompilatora:** (Wersja lub konkretny hash commita z git)
- **System operacyjny i architektura (Host):** (System, na którym uruchamiasz kompilator)
- **Target OS & Architektura (jeśli dotyczy cross-kompilacji):** (Platforma docelowa, na którą generujesz kod)
- **Użyty system budowania:** (Narzędzie zarządzające budowaniem projektu)

---

## 🛠️ Kroki do reprodukcji

### 1. Minimalny przykład kodu (MRE)
*Podaj najmniejszy możliwy fragment kodu, który wywołuje błąd. Użyj kolorowania składni.*

```python
# Tutaj wklej minimalny kod źródłowy, który psuje kompilator

```

### 2. Flagi i parametry kompilacji

*Opisz opcje, poziomy optymalizacji lub parametry, które zostały przekazane do kompilatora.*

---

## 📊 Zachowanie rzeczywiste vs Oczekiwane

### Zachowanie rzeczywiste (Actual Behavior)

*Co robi kompilator? Wklej pełne komunikaty o błędach, zrzuty stosu (stack trace) lub nieprawidłowy wynik pośredni/końcowy.*

```text
# Tutaj wklej logi błędu lub błędny output kompilatora

```

### Oczekiwane zachowanie (Expected Behavior)

*Co kompilator powinien zrobić? (np. skompilować pomyślnie, wygenerować konkretną strukturę, zwrócić czytelny błąd dla użytkownika zamiast się zawieszać).*

---

## 🔍 Dodatkowy kontekst i diagnostyka

* **Czy to regresja?** (Czy błąd występował w poprzednich wersjach Twojego kodu?)
* **Wyniki z narzędzi diagnostycznych:** (Logi z wewnętrznych warstw kompilatora, jeśli są dostępne).
* Wszelkie inne uwagi dotyczące fazy kompilacji (frontend, generowanie kodu pośredniego, optymalizacje), w której podejrzewasz problem.

