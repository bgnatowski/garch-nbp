import time
from datetime import datetime, timedelta

import pandas as pd
import requests


class NBPFetcher:
    """
    Klasa do pobierania danych kursów walutowych z API NBP
    """

    def __init__(self):
        self.base_url = 'https://api.nbp.pl/api/exchangerates/rates/'
        self.data = None

    def test_api_connection(self):
        """
        Test połączenia z API NBP - pobiera najnowszy kurs EUR
        """
        print("=== TEST POŁĄCZENIA Z API NBP ===")

        try:
            # Test z najnowszym kursem EUR z tabeli A
            url = f"{self.base_url}a/eur/last/1/?format=json"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'rates' in data and len(data['rates']) > 0:
                    print("✓ Połączenie z API NBP udane!")
                    print(f"Waluta: {data['currency']}")
                    print(f"Kod: {data['code']}")
                    print(f"Tabela: {data['table']}")

                    rate_info = data['rates'][0]
                    print(f"Najnowszy kurs EUR: {rate_info['mid']:.4f} PLN")
                    print(f"Data: {rate_info['effectiveDate']}")
                    print(f"Numer tabeli: {rate_info['no']}")

                    return True
                else:
                    print("✗ Brak danych w odpowiedzi API")
                    return False
            else:
                print(f"✗ Błąd HTTP: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Błąd połączenia: {e}")
            return False

    def get_available_currencies(self):
        """
        Pobiera listę dostępnych walut z tabeli A NBP
        """
        print("=== DOSTĘPNE WALUTY W TABELI A NBP ===")

        try:
            url = f"{self.base_url}../tables/a/last/1/?format=json"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if len(data) > 0 and 'rates' in data[0]:
                    currencies = data[0]['rates']
                    print(f"Dostępne waluty ({len(currencies)}):")
                    for curr in currencies[:10]:  # Pokazuj pierwsze 10
                        print(f"  {curr['code']}: {curr['currency']}")
                    if len(currencies) > 10:
                        print(f"  ... i {len(currencies) - 10} innych")
                    return [curr['code'] for curr in currencies]
                else:
                    print("Brak danych o walutach")
                    return []
            else:
                print(f"Błąd pobierania listy walut: {response.status_code}")
                return []

        except Exception as e:
            print(f"Błąd: {e}")
            return []

    def fetch_historical_data(self, currency_code='eur', start_date='2024-01-01',
                              end_date='2024-12-31', table='a'):
        """
        Pobieranie danych historycznych dla waluty z API NBP

        Parametry:
        currency_code (str): Kod waluty (np. 'eur', 'usd', 'gbp')
        start_date (str): Data początkowa YYYY-MM-DD
        end_date (str): Data końcowa YYYY-MM-DD
        table (str): Tabela NBP ('a', 'b', 'c') - domyślnie 'a'
        Źródło: https://api.nbp.pl
        Tabela A kursów średnich walut obcych,
        Tabela B kursów średnich walut obcych,
        Tabela C kursów kupna i sprzedaży walut obcych;
        """
        print(f"=== POBIERANIE DANYCH HISTORYCZNYCH {currency_code.upper()} ===")
        print(f"Tabela: {table.upper()}")
        print(f"Okres: {start_date} do {end_date}")

        try:
            # Konwersja dat
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # Sprawdzenie długości okresu
            total_days = (end_dt - start_dt).days + 1
            print(f"Łączna liczba dni: {total_days}")

            if total_days > 93:
                print("⚠ Okres dłuższy niż 93 dni - będzie podzielony na części (przez limit API NBP)")

            # Lista do przechowywania wszystkich danych
            all_data = []

            # Podział na okresy max 93 dni (limit API NBP)
            current_start = start_dt
            request_count = 0

            while current_start <= end_dt:
                # Określenie końca bieżącego okresu (max 93 dni)
                current_end = min(current_start + timedelta(days=92), end_dt)

                # Format dat dla API
                start_str = current_start.strftime('%Y-%m-%d')
                end_str = current_end.strftime('%Y-%m-%d')

                # Tworzenie URL dla API NBP
                url = f"{self.base_url}{table}/{currency_code}/{start_str}/{end_str}/?format=json"

                try:
                    print(f"Pobieranie: {start_str} - {end_str}")
                    response = requests.get(url, timeout=15)
                    request_count += 1

                    if response.status_code == 200:
                        data = response.json()

                        if 'rates' in data:
                            rates = data['rates']
                            all_data.extend(rates)
                            print(f"  ✓ Pobrano {len(rates)} kursów")
                        else:
                            print(f"  ⚠ Brak danych rates w odpowiedzi")

                    elif response.status_code == 404:
                        print(f"  ⚠ Brak danych dla okresu {start_str} - {end_str}")

                    else:
                        print(f"  ✗ Błąd HTTP {response.status_code}")

                except Exception as e:
                    print(f"  ✗ Błąd żądania: {e}")

                # Przejście do następnego okresu
                current_start = current_end + timedelta(days=1)

                # Krótka przerwa żeby nie przeciążyć API
                time.sleep(0.2)

            print(f"\nPodsumowanie pobierania:")
            print(f"Wykonano {request_count} żądań API")
            print(f"Pobrano łącznie {len(all_data)} kursów")

            if len(all_data) == 0:
                print("✗ Nie udało się pobrać żadnych danych")
                return False

            # Konwersja do DataFrame
            df = pd.DataFrame(all_data)

            # Konwersja daty na datetime i ustawienie jako indeks
            df['effectiveDate'] = pd.to_datetime(df['effectiveDate'])
            df.set_index('effectiveDate', inplace=True)
            df.sort_index(inplace=True)

            # Usunięcie duplikatów (na wszelki wypadek)
            df = df[~df.index.duplicated(keep='first')]

            # Przygotowanie danych w formacie kompatybilnym z GARCH
            # NBP podaje tylko kurs średni (mid)
            self.data = pd.DataFrame({
                'Close': df['mid'],  # Kurs zamknięcia = kurs średni NBP
                'Open': df['mid'],  # Open = Close (brak danych)
                'High': df['mid'] * 1.001,  # Symulowane High (+0.1%)
                'Low': df['mid'] * 0.999,  # Symulowane Low (-0.1%)
                'Volume': 1000000,  # Dummy volume
                'NBP_Rate': df['mid'],  # Oryginalny kurs NBP
                'NBP_No': df['no']  # Numer tabeli NBP
            })

            print(f"\nZakres dat: {self.data.index.min().date()} - {self.data.index.max().date()}")
            print(
                f"Zakres kursów {currency_code.upper()}: {self.data['Close'].min():.4f} - {self.data['Close'].max():.4f} PLN")
            print(f"Średni kurs: {self.data['Close'].mean():.4f} PLN")

            return True

        except Exception as e:
            print(f"✗ Błąd podczas pobierania danych: {e}")
            return False

    def save_data(self, filename=None):
        """
        Zapis danych do pliku CSV
        """
        if self.data is None:
            print("✗ Brak danych do zapisania")
            return False

        if filename is None:
            # Automatyczna nazwa pliku z datami
            start_date = self.data.index.min().strftime('%Y%m%d')
            end_date = self.data.index.max().strftime('%Y%m%d')
            filename = f"nbp_data_{start_date}_{end_date}.csv"

        try:
            self.data.to_csv(filename)
            print(f"✓ Dane zapisane do pliku: {filename}")
            return True
        except Exception as e:
            print(f"✗ Błąd zapisu: {e}")
            return False

    def load_data(self, filename):
        """
        Wczytanie danych z pliku CSV
        """
        try:
            self.data = pd.read_csv(filename, parse_dates=True, index_col=0)
            print(f"✓ Dane wczytane z pliku: {filename}")
            print(f"Liczba obserwacji: {len(self.data)}")
            print(f"Zakres dat: {self.data.index.min().date()} - {self.data.index.max().date()}")
            return True
        except FileNotFoundError:
            print(f"✗ Plik {filename} nie został znaleziony")
            return False
        except Exception as e:
            print(f"✗ Błąd wczytywania: {e}")
            return False

    def show_sample(self, n=10):
        """
        Wyświetlenie próbki danych
        """
        if self.data is None:
            print("✗ Brak danych do wyświetlenia")
            return False

        print(f"\n=== PRÓBKA DANYCH (pierwsze {n} wierszy) ===")
        print(self.data.head(n))

        print(f"\n=== STATYSTYKI PODSTAWOWE ===")
        print(self.data['Close'].describe())

        print(f"\n=== INFORMACJE O ZBIORZE ===")
        print(f"Okres: {self.data.index.min().date()} - {self.data.index.max().date()}")
        print(f"Liczba obserwacji: {len(self.data)}")
        print(f"Brakujące wartości: {self.data['Close'].isna().sum()}")

        return True

    def get_data_for_garch(self):
        """
        Zwraca dane w formacie gotowym do analizy GARCH
        """
        if self.data is not None:
            return self.data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        else:
            return None


# Funkcja główna do testowania
def main():
    """
    Funkcja główna do testowania NBP fetcher'a
    """
    print("TESTER API NBP")
    print("=" * 50)

    # Utworzenie fetcher'a
    fetcher = NBPFetcher()

    # Test połączenia
    if not fetcher.test_api_connection():
        print("Nie można nawiązać połączenia z API NBP!")
        return

    # Pokazanie dostępnych walut
    fetcher.get_available_currencies()

    print("\n" + "=" * 50)
    print("OPCJE:")
    print("1. Pobierz nowe dane historyczne")
    print("2. Wczytaj istniejące dane z pliku")
    print("3. Tylko test API")

    try:
        choice = input("\nWybierz opcję (1/2/3): ").strip()

        if choice == "1":
            # Pobieranie nowych danych
            currency = input("Kod waluty [eur]: ").strip().lower()
            if not currency:
                currency = "eur"

            start_date = input("Data początkowa (YYYY-MM-DD) [2023-01-01]: ").strip()
            if not start_date:
                start_date = "2023-01-01"

            end_date = input("Data końcowa (YYYY-MM-DD) [2024-12-31]: ").strip()
            if not end_date:
                end_date = "2024-12-31"

            table = input("Tabela NBP [a]: ").strip().lower()
            if not table:
                table = "a"

            if fetcher.fetch_historical_data(currency, start_date, end_date, table):
                fetcher.show_sample()

                save = input("\nZapisać dane do pliku? (tak/nie): ").strip().lower()
                if save in ['tak', 'yes', 'y', 't']:
                    fetcher.save_data()

        elif choice == "2":
            # Wczytanie istniejących danych
            filename = input("Nazwa pliku: ").strip()
            if filename and fetcher.load_data(filename):
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
