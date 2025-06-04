# Import wszystkich potrzebnych bibliotek
import pandas as pd  # Obsługa danych tabelarycznych
import numpy as np  # Operacje matematyczne na macierzach
import yfinance as yf  # Pobieranie danych finansowych z Yahoo Finance
import matplotlib.pyplot as plt  # Tworzenie wykresów
import seaborn as sns  # Zaawansowane wizualizacje
from arch import arch_model  # Modelowanie ARCH/GARCH
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Metryki błędów
from scipy import stats  # Testy statystyczne
import statsmodels.api as sm  # Modele statystyczne
from statsmodels.stats.diagnostic import het_breuschpagan  # Test Breuscha-Pagana
import time  # Obsługa opóźnień czasowych
import warnings  # Kontrola ostrzeżeń

warnings.filterwarnings('ignore')  # Ukrycie ostrzeżeń dla czystości outputu

# Ustawienia dla wykresów - lepszy wygląd
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GARCHAnalysis:
    """
    Klasa do kompleksowej analizy modelu GARCH dla kursów walutowych
    """

    def __init__(self, currency_pair='EURUSD=X', start_date='2020-01-01', end_date='2024-12-31', csv_file=None):
        """
        Inicjalizacja klasy z podstawowymi parametrami
        """
        self.currency_pair = currency_pair  # Para walutowa do analizy
        self.start_date = start_date  # Data początkowa pobierania danych
        self.end_date = end_date  # Data końcowa pobierania danych
        self.csv_file = csv_file  # Opcjonalny plik CSV z danymi
        self.raw_data = None  # Surowe dane cenowe
        self.returns = None  # Obliczone stopy zwrotu
        self.model = None  # Model GARCH
        self.results = None  # Wyniki dopasowania modelu

    def download_data(self):
        """
        Pobieranie danych historycznych kursu walutowego z różnych źródeł
        """
        # Jeśli podano plik CSV, próbujemy najpierw z niego
        if self.csv_file:
            try:
                print(f"Próba wczytania danych z pliku CSV: {self.csv_file}")
                self.raw_data = pd.read_csv(self.csv_file, parse_dates=True, index_col=0)
                print(f"Pomyślnie wczytano {len(self.raw_data)} obserwacji z pliku CSV")
                return
            except FileNotFoundError:
                print(f"Plik {self.csv_file} nie został znaleziony. Próba pobrania z internetu...")
            except Exception as e:
                print(f"Błąd wczytywania z CSV: {e}. Próba pobrania z internetu...")

        # Próba pobrania danych z yfinance z obsługą rate limiting
        print(f"Pobieranie danych dla pary {self.currency_pair} z Yahoo Finance...")

        max_retries = 3  # Maksymalna liczba prób
        retry_delay = 10  # Opóźnienie między próbami (sekundy)

        for attempt in range(max_retries):
            try:
                # Dodanie parametru auto_adjust=False żeby uniknąć ostrzeżenia
                self.raw_data = yf.download(self.currency_pair,
                                            start=self.start_date,
                                            end=self.end_date,
                                            progress=False,
                                            auto_adjust=False)  # Dodanie tego parametru

                # Sprawdzenie czy dane zostały pobrane poprawnie
                if not self.raw_data.empty:
                    print(f"Pomyślnie pobrano {len(self.raw_data)} obserwacji z Yahoo Finance")
                    # Zapis danych do pliku CSV jako backup
                    backup_filename = f"{self.currency_pair.replace('=', '_')}_backup.csv"
                    self.raw_data.to_csv(backup_filename)
                    print(f"Dane zapisane jako backup w pliku: {backup_filename}")
                    return
                else:
                    print(f"Próba {attempt + 1}: Pobrane dane są puste")

            except Exception as e:
                print(f"Próba {attempt + 1} nieudana: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Ponowna próba za {retry_delay} sekund...")
                    time.sleep(retry_delay)
                else:
                    print("Wszystkie próby pobierania danych z Yahoo Finance nieudane")

        # Jeśli wszystkie próby z yfinance zawiodły, próbujemy z przykładowymi danymi
        print("Generowanie przykładowych danych do demonstracji...")
        self._generate_sample_data()

    def _generate_sample_data(self):
        """
        Generowanie przykładowych danych kursu walutowego do celów demonstracyjnych
        """
        print("Tworzenie syntetycznych danych kursu EUR/USD...")

        # Tworzenie dat
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        # Symulacja random walk z trendem dla kursu walutowego
        np.random.seed(42)  # Dla reprodukowalności wyników
        n_days = len(dates)

        # Parametry symulacji
        initial_price = 1.20  # Początkowy kurs EUR/USD
        drift = 0.0001  # Niewielki trend
        volatility = 0.01  # Dzienna zmienność

        # Generowanie losowych szoków
        shocks = np.random.normal(drift, volatility, n_days)

        # Tworzenie ścieżki ceny (random walk z trendem)
        prices = [initial_price]
        for shock in shocks[1:]:
            new_price = prices[-1] * (1 + shock)
            prices.append(new_price)

        # Tworzenie DataFrame w formacie podobnym do yfinance
        self.raw_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, n_days)
        }, index=dates)

        print(f"Wygenerowano {len(self.raw_data)} syntetycznych obserwacji")
        print("UWAGA: To są przykładowe dane, nie rzeczywiste kursy walutowe!")

    def calculate_returns(self):
        """
        Obliczanie logarytmicznych stóp zwrotu z cen zamknięcia
        """
        # Logarytmiczne stopy zwrotu: log(P_t / P_{t-1}) * 100 (w procentach)
        self.returns = 100 * np.log(self.raw_data['Close'] / self.raw_data['Close'].shift(1))

        # Usunięcie pierwszej obserwacji (NaN) powstałej przez przesunięcie
        self.returns = self.returns.dropna()

        print(f"\nObliczono {len(self.returns)} stóp zwrotu")
        print("Podstawowe statystyki stóp zwrotu:")
        print(self.returns.describe())

        # Sprawdzenie własności statystycznych zwrotów
        self._analyze_returns_properties()

    def _analyze_returns_properties(self):
        """
        Analiza właściwości statystycznych stóp zwrotu
        """
        print("\n" + "=" * 50)
        print("ANALIZA WŁAŚCIWOŚCI STÓP ZWROTU")
        print("=" * 50)

        # Test Jarque-Bera na normalność rozkładu
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        print(f"Test Jarque-Bera na normalność:")
        print(f"  Statystyka: {jb_stat:.4f}")
        print(f"  P-wartość: {jb_pvalue:.4f}")
        print(f"  Rozkład normalny: {'NIE' if jb_pvalue < 0.05 else 'TAK'}")

        # Skośność i kurtoza
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        print(f"\nSkośność: {skewness:.4f}")
        print(f"Kurtoza: {kurtosis:.4f} (nadmierna kurtoza: {kurtosis:.4f})")

        # Test Breuscha-Pagana na heteroskedastyczność (zamiast ARCH test)
        self._test_heteroskedasticity_bp()

    def _test_heteroskedasticity_bp(self):
        """
        Test Breuscha-Pagana na heteroskedastyczność (zamiast testu ARCH)
        """
        print(f"\nTest Breuscha-Pagana na heteroskedastyczność:")

        try:
            # Przygotowanie danych dla regresji
            X = np.arange(len(self.returns))  # Zmienna czasowa
            X = sm.add_constant(X)  # Dodanie stałej do regresji

            # Regresja kwadratów reszt na zmienną czasową
            model = sm.OLS(self.returns ** 2, X).fit()

            # Test Breuscha-Pagana
            bp_test = het_breuschpagan(model.resid, model.model.exog)

            print(f"  Statystyka LM: {bp_test[0]:.4f}")
            print(f"  P-wartość: {bp_test[1]:.4f}")
            print(f"  Statystyka F: {bp_test[2]:.4f}")
            print(f"  P-wartość F: {bp_test[3]:.4f}")
            print(f"  Heteroskedastyczność: {'TAK' if bp_test[1] < 0.05 else 'NIE'}")

        except Exception as e:
            print(f"  Błąd podczas testu heteroskedastyczności: {e}")
            print("  Pominięto test heteroskedastyczności")

    def visualize_data(self):
        """
        Tworzenie wykresów cen i stóp zwrotu
        """
        # Utworzenie subplotów w układzie 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analiza kursu {self.currency_pair}', fontsize=16, fontweight='bold')

        # Wykres 1: Ceny zamknięcia w czasie
        axes[0, 0].plot(self.raw_data.index, self.raw_data['Close'],
                        color='blue', linewidth=1)
        axes[0, 0].set_title('Ceny zamknięcia')
        axes[0, 0].set_ylabel('Kurs')
        axes[0, 0].grid(True, alpha=0.3)

        # Wykres 2: Stopy zwrotu w czasie
        axes[0, 1].plot(self.returns.index, self.returns,
                        color='red', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_title('Logarytmiczne stopy zwrotu (%)')
        axes[0, 1].set_ylabel('Stopa zwrotu (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Wykres 3: Histogram stóp zwrotu z krzywą normalną
        axes[1, 0].hist(self.returns, bins=50, density=True, alpha=0.7,
                        color='green', edgecolor='black')

        # Dodanie krzywej rozkładu normalnego dla porównania
        mu, sigma = self.returns.mean(), self.returns.std()
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        axes[1, 0].plot(x, normal_curve, 'r--', linewidth=2,
                        label=f'Rozkład normalny\n(μ={mu:.4f}, σ={sigma:.4f})')
        axes[1, 0].set_title('Histogram stóp zwrotu')
        axes[1, 0].set_xlabel('Stopa zwrotu (%)')
        axes[1, 0].set_ylabel('Gęstość')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Wykres 4: Q-Q plot dla sprawdzenia normalności
        stats.probplot(self.returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q plot (normalność)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def fit_garch_model(self, p=1, q=1, distribution='normal'):
        """
        Dopasowanie modelu GARCH(p,q) do danych
        """
        print(f"\nDopasowywanie modelu GARCH({p},{q}) z rozkładem {distribution}...")

        try:
            # Definicja modelu GARCH z zerowymi średnimi (mean='Zero')
            self.model = arch_model(self.returns,
                                    mean='Zero',  # Zerowa średnia warunkowa
                                    vol='GARCH',  # Model GARCH dla wariancji
                                    p=p,  # Opóźnienia wariancji (GARCH)
                                    q=q,  # Opóźnienia kwadratów reszt (ARCH)
                                    dist=distribution)  # Rozkład składnika losowego

            # Dopasowanie modelu metodą maksymalnej wiarygodności
            self.results = self.model.fit(disp='off')  # disp='off' wyłącza szczegółowe komunikaty

            # Wyświetlenie podsumowania wyników
            print(self.results.summary())

            # Analiza parametrów modelu
            self._analyze_model_parameters()

        except Exception as e:
            print(f"Błąd podczas dopasowywania modelu GARCH: {e}")
            raise

    def _analyze_model_parameters(self):
        """
        Analiza i interpretacja oszacowanych parametrów modelu GARCH
        """
        params = self.results.params

        print("\n" + "=" * 50)
        print("ANALIZA PARAMETRÓW MODELU GARCH(1,1)")
        print("=" * 50)

        # Wyciągnięcie parametrów (nazwy mogą się różnić w zależności od wersji biblioteki)
        omega = params['omega']  # Stała w równaniu wariancji
        alpha = params['alpha[1]']  # Parametr ARCH
        beta = params['beta[1]']  # Parametr GARCH

        print(f"ω (omega) = {omega:.6f}")
        print(f"α (alpha) = {alpha:.6f}")
        print(f"β (beta)  = {beta:.6f}")
        print(f"α + β = {alpha + beta:.6f}")

        # Sprawdzenie warunków stacjonarności
        if alpha + beta < 1:
            print("✓ Warunek stacjonarności spełniony (α + β < 1)")

            # Długookresowa (bezwarunkowa) wariancja
            unconditional_var = omega / (1 - alpha - beta)
            unconditional_vol = np.sqrt(unconditional_var)
            print(f"Długookresowa wariancja: {unconditional_var:.6f}")
            print(f"Długookresowa zmienność: {unconditional_vol:.4f}%")
        else:
            print("⚠ Ostrzeżenie: Warunek stacjonarności niespełniony!")

        # Interpretacja parametrów
        print(f"\nInterpretacja parametrów:")
        print(f"- α = {alpha:.4f}: Siła reakcji na ostatni szok")
        print(f"- β = {beta:.4f}: Trwałość zmienności")
        print(f"- Połowa życia szoku: {np.log(0.5) / np.log(alpha + beta):.1f} dni")

    def forecast_volatility(self, horizon=30):
        """
        Prognozowanie zmienności na zadany horyzont czasowy
        """
        print(f"\nGenerowanie prognoz zmienności na {horizon} dni...")

        try:
            # Generowanie prognoz używając dopasowanego modelu
            forecasts = self.results.forecast(horizon=horizon)

            # Wyciągnięcie prognozowanych wariancji i przekształcenie na zmienność (%)
            forecast_variance = forecasts.variance.values[-1, :]  # Ostatni wiersz zawiera prognozy
            forecast_volatility = np.sqrt(forecast_variance)  # Zmienność = sqrt(wariancja)

            # Utworzenie dat dla prognoz (kontynuacja od ostatniej daty w danych)
            last_date = self.returns.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=horizon, freq='D')

            # Tworzenie DataFrame z prognozami
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast_volatility': forecast_volatility
            })

            print("Pierwsze 10 prognoz zmienności:")
            print(forecast_df.head(10))

            # Wizualizacja prognoz
            self._plot_volatility_forecast(forecast_df)

            return forecast_df

        except Exception as e:
            print(f"Błąd podczas prognozowania: {e}")
            return pd.DataFrame()  # Zwróć pusty DataFrame w przypadku błędu

    def _plot_volatility_forecast(self, forecast_df):
        """
        Wizualizacja historycznej i prognozowanej zmienności
        """
        try:
            # Obliczenie historycznej zmienności (rolling std na oknie 30 dni)
            historical_vol = self.returns.rolling(window=30).std()

            # Utworzenie wykresu
            plt.figure(figsize=(15, 8))

            # Wykres historycznej zmienności (ostatnie 200 obserwacji dla czytelności)
            plt.plot(historical_vol.index[-200:], historical_vol.iloc[-200:],
                     label='Historyczna zmienność (30-dniowa)', color='blue', linewidth=1.5)

            # Wykres prognozowanej zmienności
            if not forecast_df.empty:
                plt.plot(forecast_df['date'], forecast_df['forecast_volatility'],
                         label='Prognoza GARCH', color='red', linewidth=2, linestyle='--')

            # Dodanie linii pionowej oznaczającej początek prognozy
            plt.axvline(x=self.returns.index[-1], color='green', linestyle=':',
                        label='Początek prognozy', alpha=0.7)

            plt.title('Historyczna i prognozowana zmienność kursu walutowego',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('Zmienność (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Błąd podczas tworzenia wykresu prognoz: {e}")

    def validate_model(self, test_size=0.2):
        """
        Walidacja modelu przez podział na zbiór treningowy i testowy
        """
        print(f"\nWalidacja modelu (test_size = {test_size})...")

        try:
            # Podział danych na zbiór treningowy i testowy
            split_point = int(len(self.returns) * (1 - test_size))
            train_returns = self.returns.iloc[:split_point]
            test_returns = self.returns.iloc[split_point:]

            print(f"Zbiór treningowy: {len(train_returns)} obserwacji")
            print(f"Zbiór testowy: {len(test_returns)} obserwacji")

            # Dopasowanie modelu na zbiorze treningowym
            train_model = arch_model(train_returns, mean='Zero', vol='GARCH', p=1, q=1)
            train_results = train_model.fit(disp='off')

            # Prognozowanie na zbiorze testowym (uproszczona wersja)
            forecasted_volatilities = []
            actual_volatilities = []

            print("Generowanie prognoz rolling...")

            # Uproszczona walidacja - prognoza z jednego modelu
            forecasts = train_results.forecast(horizon=len(test_returns))
            forecast_variance = forecasts.variance.values[-1, :]
            forecasted_volatilities = np.sqrt(forecast_variance)

            # Rzeczywista zmienność (absolutna wartość zwrotu jako proxy)
            actual_volatilities = np.abs(test_returns.values)

            # Dopasowanie długości (na wypadek różnic)
            min_length = min(len(forecasted_volatilities), len(actual_volatilities))
            forecasted_volatilities = forecasted_volatilities[:min_length]
            actual_volatilities = actual_volatilities[:min_length]

            # Obliczenie metryk błędów
            rmse = np.sqrt(mean_squared_error(actual_volatilities, forecasted_volatilities))
            mae = mean_absolute_error(actual_volatilities, forecasted_volatilities)

            # Bezpieczne obliczenie MAPE (unikanie dzielenia przez zero)
            non_zero_actual = actual_volatilities[actual_volatilities != 0]
            non_zero_forecast = forecasted_volatilities[actual_volatilities != 0]

            if len(non_zero_actual) > 0:
                mape = np.mean(np.abs((non_zero_actual - non_zero_forecast) / non_zero_actual)) * 100
            else:
                mape = float('inf')

            print(f"\nMetryki jakości prognoz:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")

            # Wizualizacja porównania prognoz z rzeczywistością
            self._plot_validation_results(test_returns.index[:min_length],
                                          actual_volatilities, forecasted_volatilities)

            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'actual': actual_volatilities,
                'predicted': forecasted_volatilities
            }

        except Exception as e:
            print(f"Błąd podczas walidacji modelu: {e}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'actual': [],
                'predicted': []
            }

    def _plot_validation_results(self, dates, actual, predicted):
        """
        Wizualizacja wyników walidacji - porównanie prognoz z rzeczywistością
        """
        try:
            plt.figure(figsize=(15, 10))

            # Górny subplot: Porównanie szeregów czasowych
            plt.subplot(2, 1, 1)
            plt.plot(dates, actual, label='Rzeczywista zmienność',
                     color='blue', alpha=0.7, linewidth=1)
            plt.plot(dates, predicted, label='Prognoza GARCH',
                     color='red', alpha=0.8, linewidth=1.5)
            plt.title('Porównanie rzeczywistej i prognozowanej zmienności')
            plt.ylabel('Zmienność (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Dolny subplot: Wykres rozrzutu (scatter plot)
            plt.subplot(2, 1, 2)
            plt.scatter(actual, predicted, alpha=0.6, color='green')

            # Linia 45 stopni (idealna prognoza)
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                     label='Idealna prognoza', linewidth=2)

            plt.xlabel('Rzeczywista zmienność (%)')
            plt.ylabel('Prognozowana zmienność (%)')
            plt.title('Wykres rozrzutu: Prognoza vs Rzeczywistość')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Błąd podczas tworzenia wykresów walidacji: {e}")

    def run_complete_analysis(self):
        """
        Uruchomienie kompletnej analizy GARCH - główna funkcja orkiestrująca
        """
        print("=" * 60)
        print("ROZPOCZĘCIE KOMPLETNEJ ANALIZY GARCH")
        print("=" * 60)

        try:
            # Krok 1: Pobranie danych
            self.download_data()

            # Krok 2: Obliczenie stóp zwrotu
            self.calculate_returns()

            # Krok 3: Wizualizacja danych
            self.visualize_data()

            # Krok 4: Dopasowanie modelu GARCH
            self.fit_garch_model(p=1, q=1, distribution='normal')

            # Krok 5: Prognozowanie zmienności
            forecasts = self.forecast_volatility(horizon=30)

            # Krok 6: Walidacja modelu
            validation_results = self.validate_model(test_size=0.2)

            print("\n" + "=" * 60)
            print("ANALIZA ZAKOŃCZONA POMYŚLNIE")
            print("=" * 60)

            return {
                'model_results': self.results,
                'forecasts': forecasts,
                'validation': validation_results
            }

        except Exception as e:
            print(f"Błąd podczas analizy: {str(e)}")
            print("Sprawdź połączenie internetowe lub użyj pliku CSV z danymi")
            return None


# Główna część skryptu - wykonanie analizy
if __name__ == "__main__":
    # Utworzenie instancji klasy analizy GARCH z opcją pliku CSV
    # Można podać plik CSV jako alternatywę dla Yahoo Finance
    garch_analyzer = GARCHAnalysis(
        currency_pair='EURUSD=X',  # Para walutowa EUR/USD
        start_date='2020-01-01',  # Data początkowa
        end_date='2024-12-31',  # Data końcowa
        csv_file=None  # Opcjonalnie: 'EURUSD_data.csv'
    )

    # Uruchomienie kompletnej analizy
    results = garch_analyzer.run_complete_analysis()

    # Sprawdzenie czy analiza się powiodła
    if results is not None:
        # Opcjonalnie: Zapis wyników do pliku
        print("\nCzy chcesz zapisać wyniki do pliku CSV? (tak/nie)")
        try:
            response = input().lower().strip()

            if response in ['tak', 'yes', 'y', 't']:
                # Zapis prognoz do pliku CSV
                if not results['forecasts'].empty:
                    results['forecasts'].to_csv('garch_volatility_forecast.csv', index=False)
                    print("Prognozy zapisane do pliku: garch_volatility_forecast.csv")

                # Zapis parametrów modelu do pliku tekstowego
                if results['model_results'] is not None:
                    with open('garch_model_summary.txt', 'w', encoding='utf-8') as f:
                        f.write("PODSUMOWANIE MODELU GARCH\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(str(results['model_results'].summary()))

                    print("Podsumowanie modelu zapisane do pliku: garch_model_summary.txt")
        except EOFError:
            print("Brak inputu użytkownika - pomijam zapis plików")

    print("\nSkrypt zakończony. Dziękujemy za skorzystanie z analizy GARCH!")
