import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json


class CurrencyLayerFetcher:
    """
    Klasa do pobierania danych kursów walutowych z currencylayer.com API
    """

    def __init__(self, api_key):
        """
        Inicjalizacja z kluczem API
        """
        self.api_key = api_key
        self.base_url = "https://api.currencylayer.com/"
        self.data = None

    def test_api_connection(self):
        """
        Test połączenia z API - pobiera najnowsze kursy
        """
        print("=== TEST POŁĄCZENIA Z CURRENCYLAYER API ===")

        try:
            url = f"{self.base_url}live"
            params = {
                'access_key': self.api_key,
                'currencies': 'PLN,EUR',  # Tylko PLN i EUR żeby oszczędzić API calls
                'format': 1
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('success', False):
                    print("✓ Połączenie z API udane!")
                    print(f"Timestamp: {data.get('timestamp')}")
                    print(f"Source currency: {data.get('source')}")

                    quotes = data.get('quotes', {})
                    if 'USDPLN' in quotes:
                        print(f"USD/PLN: {quotes['USDPLN']:.6f}")
                    if 'USDEUR' in quotes:
                        print(f"USD/EUR: {quotes['USDEUR']:.6f}")

                    # Obliczenie PLN/EUR
                    if 'USDPLN' in quotes and 'USDEUR' in quotes:
                        pln_eur = quotes['USDEUR'] / quotes['USDPLN']
                        print(f"PLN/EUR (obliczone): {pln_eur:.6f}")

                    return True
                else:
                    error = data.get('error', {})
                    print(f"✗ Błąd API: {error.get('info', 'Nieznany błąd')}")
                    print(f"Kod błędu: {error.get('code', 'N/A')}")
                    return False
            else:
                print(f"✗ Błąd HTTP: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Błąd połączenia: {e}")
            return False

    def fetch_historical_data(self, start_date='2024-01-01', end_date='2024-12-31',
                              base_currency='PLN', target_currency='EUR'):
        """
        Pobieranie danych historycznych dla pary walutowej

        Uwaga: currencylayer używa USD jako base, więc dla PLN/EUR pobieramy USD/PLN i USD/EUR
        """
        print(f"=== POBIERANIE DANYCH HISTORYCZNYCH {base_currency}/{target_currency} ===")
        print(f"Okres: {start_date} do {end_date}")
        print("UWAGA: To może zużyć dużo API calls - monitoruj swój limit!")

        try:
            # Konwersja dat
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # Sprawdzenie liczby dni (max 100 API calls na darmowym planie)
            total_days = (end_dt - start_dt).days + 1
            print(f"Liczba dni do pobrania: {total_days}")

            if total_days > 90:  # Zostaw trochę API calls na testy
                print("⚠ UWAGA: To może przekroczyć limit darmowego planu!")
                response = input("Kontynuować? (tak/nie): ").lower().strip()
                if response not in ['tak', 'yes', 'y', 't']:
                    print("Anulowano pobieranie danych")
                    return False

            # Listy do przechowywania danych
            dates = []
            usd_pln_rates = []
            usd_eur_rates = []

            # Licznik API calls
            api_calls_used = 0

            # Iteracja przez każdy dzień
            current_date = start_dt
            while current_date <= end_dt:
                date_str = current_date.strftime('%Y-%m-%d')

                try:
                    # URL dla danych historycznych
                    url = f"{self.base_url}historical"
                    params = {
                        'access_key': self.api_key,
                        'date': date_str,
                        'currencies': f'{base_currency},{target_currency}',  # PLN,EUR
                        'format': 1
                    }

                    response = requests.get(url, params=params, timeout=10)
                    api_calls_used += 1

                    if response.status_code == 200:
                        data = response.json()

                        if data.get('success', False):
                            quotes = data.get('quotes', {})

                            # Pobieramy USD/PLN i USD/EUR
                            usd_pln_key = f'USD{base_currency}'
                            usd_eur_key = f'USD{target_currency}'

                            if usd_pln_key in quotes and usd_eur_key in quotes:
                                dates.append(current_date)
                                usd_pln_rates.append(quotes[usd_pln_key])
                                usd_eur_rates.append(quotes[usd_eur_key])

                                # Postęp co 10 dni
                                if len(dates) % 10 == 0:
                                    print(f"Pobrano {len(dates)} obserwacji... (API calls: {api_calls_used})")
                            else:
                                print(f"Brak danych dla {date_str}")
                        else:
                            error = data.get('error', {})
                            print(f"Błąd API dla {date_str}: {error.get('info', 'Nieznany błąd')}")

                            # Jeśli przekroczono limit, przerwij
                            if error.get('code') == 104:
                                print("⚠ Przekroczono limit API calls!")
                                break
                    else:
                        print(f"Błąd HTTP {response.status_code} dla {date_str}")

                    # Krótka przerwa żeby nie przeciążyć API
                    time.sleep(0.5)

                except Exception as e:
                    print(f"Błąd pobierania danych dla {date_str}: {e}")

                # Przejście do następnego dnia
                current_date += timedelta(days=1)

            print(f"\nPodsumowanie pobierania:")
            print(f"Użyte API calls: {api_calls_used}")
            print(f"Pobrano {len(dates)} obserwacji")

            if len(dates) > 0:
                # Obliczenie kursów PLN/EUR z USD/PLN i USD/EUR
                pln_eur_rates = [eur_rate / pln_rate for eur_rate, pln_rate in zip(usd_eur_rates, usd_pln_rates)]

                # Utworzenie DataFrame
                self.data = pd.DataFrame({
                    'Date': dates,
                    'USD_PLN': usd_pln_rates,
                    'USD_EUR': usd_eur_rates,
                    'PLN_EUR_Rate': pln_eur_rates
                })

                # Ustawienie daty jako indeks
                self.data.set_index('Date', inplace=True)

                # Dodanie kolumn w formacie podobnym do yfinance
                self.data['Open'] = self.data['PLN_EUR_Rate']
                self.data['High'] = self.data['PLN_EUR_Rate'] * 1.001  # Symulowane
                self.data['Low'] = self.data['PLN_EUR_Rate'] * 0.999  # Symulowane
                self.data['Close'] = self.data['PLN_EUR_Rate']
                self.data['Volume'] = 1000000  # Dummy volume

                print(f"\nZakres dat: {self.data.index.min().date()} - {self.data.index.max().date()}")
                print(
                    f"Zakres kursów PLN/EUR: {self.data['PLN_EUR_Rate'].min():.6f} - {self.data['PLN_EUR_Rate'].max():.6f}")

                return True
            else:
                print("✗ Nie udało się pobrać żadnych danych")
                return False

        except Exception as e:
            print(f"✗ Błąd podczas pobierania danych: {e}")
            return False

    def save_data(self, filename='currencylayer_pln_eur_data.csv'):
        """
        Zapis danych do pliku CSV
        """
        if self.data is not None:
            self.data.to_csv(filename)
            print(f"✓ Dane zapisane do pliku: {filename}")
            return True
        else:
            print("✗ Brak danych do zapisania")
            return False

    def load_data(self, filename='currencylayer_pln_eur_data.csv'):
        """
        Wczytanie danych z pliku CSV
        """
        try:
            self.data = pd.read_csv(filename, parse_dates=True, index_col=0)
            print(f"✓ Dane wczytane z pliku: {filename}")
            print(f"Liczba obserwacji: {len(self.data)}")
            return True
        except FileNotFoundError:
            print(f"✗ Plik {filename} nie został znaleziony")
            return False
        except Exception as e:
            print(f"✗ Błąd wczytywania pliku: {e}")
            return False

    def show_sample(self, n=10):
        """
        Wyświetlenie próbki danych
        """
        if self.data is not None:
            print(f"\n=== PRÓBKA DANYCH (pierwsze {n} wierszy) ===")
            print(self.data.head(n))

            print(f"\n=== STATYSTYKI PODSTAWOWE ===")
            print(self.data['PLN_EUR_Rate'].describe())
            return True
        else:
            print("✗ Brak danych do wyświetlenia")
            return False

    def get_data_for_garch(self):
        """
        Zwraca dane w formacie gotowym do analizy GARCH
        """
        if self.data is not None:
            return self.data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        else:
            return None


# Funkcja testowa
def main():
    """
    Funkcja główna do testowania fetcher'a
    """
    print("TESTER CURRENCYLAYER API")
    print("=" * 50)

    # WSTAW TUTAJ SWÓJ API KEY!
    api_key = "3d2a79ecdb2d13a2832713b6fbffb9a9"

    # Utworzenie fetcher'a
    fetcher = CurrencyLayerFetcher(api_key)

    # Test połączenia
    if not fetcher.test_api_connection():
        print("Nie można nawiązać połączenia z API!")
        return

    print("\n" + "=" * 50)
    print("OPCJE:")
    print("1. Pobierz nowe dane historyczne")
    print("2. Wczytaj istniejące dane z pliku")
    print("3. Tylko test API (bez pobierania danych)")

    try:
        choice = input("\nWybierz opcję (1/2/3): ").strip()

        if choice == "1":
            # Pobieranie nowych danych
            start_date = input("Data początkowa (YYYY-MM-DD) [2024-01-01]: ").strip()
            if not start_date:
                start_date = "2024-01-01"

            end_date = input("Data końcowa (YYYY-MM-DD) [2024-03-31]: ").strip()
            if not end_date:
                end_date = "2024-03-31"  # Ograniczamy żeby nie przekroczyć 100 API calls

            if fetcher.fetch_historical_data(start_date, end_date):
                fetcher.show_sample()
                fetcher.save_data()

        elif choice == "2":
            # Wczytanie istniejących danych
            filename = input("Nazwa pliku [currencylayer_pln_eur_data.csv]: ").strip()
            if not filename:
                filename = "currencylayer_pln_eur_data.csv"

            if fetcher.load_data(filename):
                fetcher.show_sample()

        elif choice == "3":
            print("Test API zakończony.")

        else:
            print("Nieprawidłowa opcja")

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"Błąd: {e}")


if __name__ == "__main__":
    main()
