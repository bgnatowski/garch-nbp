import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json


class ForexDataFetcher:
    """
    Klasa do pobierania danych kursów walutowych z różnych darmowych API
    """

    def __init__(self):
        self.data = None

    def test_exchangeratesapi(self, start_date='2024-01-01', end_date='2024-01-31'):
        """
        Test API exchangeratesapi.io - darmowe, oparte na danych ECB
        """
        print("=== TEST exchangeratesapi.io ===")

        try:
            # URL dla najnowszego kursu
            url_latest = "https://api.exchangerate-api.com/v4/latest/PLN"

            print("Pobieranie najnowszego kursu PLN...")
            response = requests.get(url_latest, timeout=10)

            if response.status_code == 200:
                data = response.json()
                eur_rate = data['rates']['EUR']
                print(f"✓ Najnowszy kurs PLN/EUR: {eur_rate:.6f}")
                print(f"Data: {data['date']}")
                print(f"Baza: {data['base']}")
                return True
            else:
                print(f"✗ Błąd HTTP: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Błąd: {e}")
            return False

    def test_exchangeratesapi_historical(self, days_back=30):
        """
        Test pobierania danych historycznych z exchangerate-api.com
        """
        print(f"\n=== TEST danych historycznych (ostatnie {days_back} dni) ===")

        try:
            dates = []
            rates = []

            # Pobieranie danych dla ostatnich X dni
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')

                url = f"https://api.exchangerate-api.com/v4/history/PLN/{date_str}"

                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if 'rates' in data and 'EUR' in data['rates']:
                            dates.append(date)
                            rates.append(data['rates']['EUR'])

                            if len(dates) % 5 == 0:
                                print(f"Pobrano {len(dates)} obserwacji...")

                    # Krótka przerwa żeby nie przeciążyć API
                    time.sleep(0.2)

                except requests.exceptions.Timeout:
                    print(f"Timeout dla daty {date_str}")
                except Exception as e:
                    print(f"Błąd dla daty {date_str}: {e}")

            if len(dates) > 0:
                print(f"✓ Pobrano {len(dates)} kursów historycznych")
                print(f"Zakres dat: {min(dates).date()} - {max(dates).date()}")
                print(f"Zakres kursów: {min(rates):.6f} - {max(rates):.6f}")

                # Utworzenie DataFrame
                df = pd.DataFrame({
                    'Date': dates,
                    'PLN_EUR_Rate': rates
                }).sort_values('Date')

                self.data = df
                return True
            else:
                print("✗ Nie udało się pobrać żadnych danych")
                return False

        except Exception as e:
            print(f"✗ Błąd: {e}")
            return False

    def test_fixer_api(self):
        """
        Test fixer.io API (wymaga rejestracji na darmowy klucz)
        """
        print("\n=== TEST fixer.io API ===")
        print("UWAGA: Wymaga darmowej rejestracji na https://fixer.io/")

        # Przykładowy URL (wymagany klucz API)
        # api_key = "YOUR_FREE_API_KEY"  # Trzeba się zarejestrować
        # url = f"http://data.fixer.io/api/latest?access_key={api_key}&base=PLN&symbols=EUR"

        print("Pomiń ten test jeśli nie masz klucza API")
        return False

    def test_currencylayer(self):
        """
        Test currencylayer API (wymaga darmowej rejestracji)
        """
        print("\n=== TEST currencylayer API ===")
        print("UWAGA: Wymaga darmowej rejestracji na https://currencylayer.com/")

        # Przykładowy URL (wymagany klucz API)
        # api_key = "YOUR_FREE_API_KEY"
        # url = f"http://api.currencylayer.com/live?access_key={api_key}&source=PLN&currencies=EUR"

        print("Pomiń ten test jeśli nie masz klucza API")
        return False

    def test_european_central_bank(self):
        """
        Test bezpośredniego API Europejskiego Banku Centralnego
        """
        print("\n=== TEST European Central Bank API ===")

        try:
            # ECB publikuje kursy w formacie XML
            url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                print("✓ Połączenie z ECB udane")

                # Proste parsowanie XML (bez biblioteki xml)
                content = response.text
                if 'PLN' in content:
                    # Wyciągnięcie kursu PLN (bardzo proste parsowanie)
                    start = content.find("currency='PLN'")
                    if start != -1:
                        rate_start = content.find("rate='", start) + 6
                        rate_end = content.find("'", rate_start)
                        pln_rate = float(content[rate_start:rate_end])

                        # ECB podaje ile PLN za 1 EUR, my chcemy ile EUR za 1 PLN
                        eur_per_pln = 1.0 / pln_rate

                        print(f"✓ Kurs EUR/PLN: {pln_rate:.6f}")
                        print(f"✓ Kurs PLN/EUR: {eur_per_pln:.6f}")
                        return True

                print("✗ Nie znaleziono kursu PLN w danych ECB")
                return False
            else:
                print(f"✗ Błąd HTTP: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Błąd: {e}")
            return False

    def generate_sample_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        Generowanie przykładowych danych PLN/EUR z realistycznymi parametrami
        """
        print(f"\n=== GENEROWANIE SYNTETYCZNYCH DANYCH {start_date} - {end_date} ===")

        # Tworzenie dat (dni robocze)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        np.random.seed(42)  # Dla reprodukowalności
        n_days = len(dates)

        # Realistyczne parametry dla PLN/EUR
        initial_rate = 0.22  # Początkowy kurs
        annual_drift = -0.01  # Roczny trend (lekka deprecjacja PLN)
        annual_volatility = 0.12  # Roczna zmienność (12%)

        # Przeliczenie na parametry dzienne
        daily_drift = annual_drift / 252
        daily_vol = annual_volatility / np.sqrt(252)

        # Dodanie cykliczności i trendów
        trend = np.linspace(0, annual_drift, n_days)
        seasonal = 0.01 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Sezonowość roczna

        # Generowanie zmienności w stylu GARCH
        shocks = []
        vol_series = [daily_vol]

        for i in range(n_days):
            # GARCH(1,1) dla zmiennej zmienności
            current_vol = np.sqrt(0.000001 + 0.08 * (shocks[-1] ** 2 if shocks else 0) + 0.9 * vol_series[-1] ** 2)
            vol_series.append(current_vol)

            shock = np.random.normal(daily_drift, current_vol)
            shocks.append(shock)

        # Budowanie ścieżki kursu
        rates = [initial_rate]
        for i, shock in enumerate(shocks):
            new_rate = rates[-1] * np.exp(shock + trend[i] + seasonal[i])
            rates.append(new_rate)

        rates = rates[1:]  # Usunięcie pierwszej wartości

        # Utworzenie DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'PLN_EUR_Rate': rates,
            'Daily_Return': [0] + [np.log(rates[i] / rates[i - 1]) * 100 for i in range(1, len(rates))],
            'Volatility': vol_series[1:n_days + 1]
        })

        print(f"✓ Wygenerowano {len(df)} obserwacji")
        print(f"Zakres dat: {df['Date'].min().date()} - {df['Date'].max().date()}")
        print(f"Zakres kursów: {df['PLN_EUR_Rate'].min():.6f} - {df['PLN_EUR_Rate'].max():.6f}")
        print(f"Średnia dzienna zmiana: {df['Daily_Return'].mean():.4f}%")
        print(f"Dzienna zmienność: {df['Daily_Return'].std():.4f}%")
        print(f"Roczna zmienność: {df['Daily_Return'].std() * np.sqrt(252):.2f}%")

        self.data = df
        return True

    def save_data(self, filename='pln_eur_data.csv'):
        """
        Zapis danych do pliku CSV
        """
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            print(f"\n✓ Dane zapisane do pliku: {filename}")
            return True
        else:
            print("\n✗ Brak danych do zapisania")
            return False

    def show_sample(self, n=10):
        """
        Wyświetlenie próbki danych
        """
        if self.data is not None:
            print(f"\n=== PRÓBKA DANYCH (pierwsze {n} wierszy) ===")
            print(self.data.head(n))
            return True
        else:
            print("\n✗ Brak danych do wyświetlenia")
            return False


def main():
    """
    Główna funkcja testująca różne źródła danych forex
    """
    print("TESTER ŹRÓDEŁ DANYCH FOREX PLN/EUR")
    print("=" * 50)

    fetcher = ForexDataFetcher()

    # Test 1: exchangerate-api.com (najprostsze, bez rejestracji)
    success1 = fetcher.test_exchangeratesapi()

    # Test 2: Dane historyczne
    success2 = fetcher.test_exchangeratesapi_historical(days_back=10)

    # Test 3: ECB (oficjalne źródło)
    success3 = fetcher.test_european_central_bank()

    # Test 4: Inne API (wymagają rejestracji)
    fetcher.test_fixer_api()
    fetcher.test_currencylayer()

    # Jeśli żadne API nie działa, generuj dane syntetyczne
    if not (success1 or success2 or success3):
        print("\n" + "!" * 50)
        print("WSZYSTKIE API NIEDOSTĘPNE - GENEROWANIE DANYCH SYNTETYCZNYCH")
        print("!" * 50)
        fetcher.generate_sample_data()

    # Wyświetl próbkę danych
    fetcher.show_sample()

    # Zapisz dane
    fetcher.save_data()

    print("\n" + "=" * 50)
    print("TEST ZAKOŃCZONY")
    print("=" * 50)

    # Podsumowanie
    if fetcher.data is not None:
        print("✓ Dane dostępne do analizy GARCH")
        print("Możesz teraz użyć pliku 'pln_eur_data.csv' w głównym skrypcie")
    else:
        print("✗ Brak danych - sprawdź połączenie internetowe")


if __name__ == "__main__":
    main()
