# Import wszystkich potrzebnych bibliotek
import pandas as pd  # Obsługa danych tabelarycznych
import numpy as np  # Operacje matematyczne na macierzach
import matplotlib.pyplot as plt  # Tworzenie wykresów
import seaborn as sns  # Zaawansowane wizualizacje
from arch import arch_model  # Modelowanie ARCH/GARCH
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Metryki błędów
from scipy import stats  # Testy statystyczne
import statsmodels.api as sm  # Modele statystyczne
from statsmodels.stats.diagnostic import het_breuschpagan  # Test Breuscha-Pagana
import warnings  # Kontrola ostrzeżeń

# Import naszego fetcher'a
from currencylayer_fetcher import CurrencyLayerFetcher

warnings.filterwarnings('ignore')  # Ukrycie ostrzeżeń dla czystości outputu

# Ustawienia dla wykresów - lepszy wygląd
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GARCHAnalysisCurrencyLayer:
    """
    Klasa do kompleksowej analizy modelu GARCH dla kursów walutowych PLN/EUR
    z wykorzystaniem danych z CurrencyLayer API
    """

    def __init__(self, api_key=None, base_currency='PLN', target_currency='EUR',
                 start_date='2025-03-01', end_date='2025-03-15', csv_file=None):
        """
        Inicjalizacja klasy z podstawowymi parametrami
        """
        self.api_key = api_key  # Klucz API CurrencyLayer
        self.base_currency = base_currency  # Waluta bazowa (PLN)
        self.target_currency = target_currency  # Waluta docelowa (EUR)
        self.currency_pair = f"{base_currency}/{target_currency}"
        self.start_date = start_date  # Data początkowa pobierania danych
        self.end_date = end_date  # Data końcowa pobierania danych
        self.csv_file = csv_file  # Opcjonalny plik CSV z danymi
        self.raw_data = None  # Surowe dane cenowe
        self.returns = None  # Obliczone stopy zwrotu
        self.model = None  # Model GARCH
        self.results = None  # Wyniki dopasowania modelu
        self.fetcher = None  # Obiekt fetcher'a

    def setup_data_source(self):
        """
        Konfiguracja źródła danych - API lub plik CSV
        """
        print("=== KONFIGURACJA ŹRÓDŁA DANYCH ===")

        # Jeśli podano plik CSV, próbujemy z niego
        if self.csv_file:
            print(f"Próba wczytania danych z pliku CSV: {self.csv_file}")
            try:
                self.raw_data = pd.read_csv(self.csv_file, parse_dates=True, index_col=0)

                # Sprawdzenie czy plik ma odpowiednie kolumny
                required_columns = ['Close']
                if not all(col in self.raw_data.columns for col in required_columns):
                    # Jeśli nie ma kolumny Close, użyj PLN_EUR_Rate
                    if 'PLN_EUR_Rate' in self.raw_data.columns:
                        self.raw_data['Close'] = self.raw_data['PLN_EUR_Rate']
                        self.raw_data['Open'] = self.raw_data['PLN_EUR_Rate']
                        self.raw_data['High'] = self.raw_data['PLN_EUR_Rate'] * 1.001
                        self.raw_data['Low'] = self.raw_data['PLN_EUR_Rate'] * 0.999
                        self.raw_data['Volume'] = 1000000

                print(f"✓ Wczytano {len(self.raw_data)} obserwacji z pliku CSV")
                return True

            except Exception as e:
                print(f"✗ Błąd wczytywania CSV: {e}")

        # Jeśli nie ma pliku lub się nie udał, użyj API
        if self.api_key:
            print("Próba pobrania danych z CurrencyLayer API...")
            self.fetcher = CurrencyLayerFetcher(self.api_key)

            # Test połączenia z API
            if not self.fetcher.test_api_connection():
                print("✗ Nie można nawiązać połączenia z CurrencyLayer API")
                return False

            # Pobieranie danych historycznych
            if self.fetcher.fetch_historical_data(
                    self.start_date, self.end_date,
                    self.base_currency, self.target_currency
            ):
                self.raw_data = self.fetcher.get_data_for_garch()
                print("✓ Dane pobrane z CurrencyLayer API")

                # Automatyczny zapis jako backup
                backup_filename = f"currencylayer_{self.base_currency}_{self.target_currency}_backup.csv"
                self.fetcher.save_data(backup_filename)

                return True
            else:
                print("✗ Nie udało się pobrać danych z API")
                return False

        print("✗ Brak dostępnych źródeł danych (ani CSV ani API key)")
        return False

    def calculate_returns(self):
        """
        Obliczanie logarytmicznych stóp zwrotu z cen zamknięcia
        """
        if self.raw_data is None:
            print("✗ Brak danych do obliczenia stóp zwrotu")
            return False

        # Logarytmiczne stopy zwrotu: log(P_t / P_{t-1}) * 100 (w procentach)
        self.returns = 100 * np.log(self.raw_data['Close'] / self.raw_data['Close'].shift(1))

        # Usunięcie pierwszej obserwacji (NaN) powstałej przez przesunięcie
        self.returns = self.returns.dropna()

        print(f"\n✓ Obliczono {len(self.returns)} stóp zwrotu dla pary {self.currency_pair}")
        print("Podstawowe statystyki stóp zwrotu:")
        print(self.returns.describe())

        # Sprawdzenie własności statystycznych zwrotów
        self._analyze_returns_properties()
        return True

    def _analyze_returns_properties(self):
        """
        Analiza właściwości statystycznych stóp zwrotu
        """
        print("\n" + "=" * 50)
        print(f"ANALIZA WŁAŚCIWOŚCI STÓP ZWROTU {self.currency_pair}")
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

        # Test Breuscha-Pagana na heteroskedastyczność
        self._test_heteroskedasticity_bp()

        # Dodatkowe statystyki dla waluty
        print(f"\nDodatkowe statystyki dla {self.currency_pair} (CurrencyLayer API):")
        print(f"Średnia dzienna zmiana: {self.returns.mean():.4f}%")
        print(f"Dzienna zmienność: {self.returns.std():.4f}%")
        print(f"Roczna zmienność: {self.returns.std() * np.sqrt(252):.2f}%")
        print(f"Maksymalna dzienna strata: {self.returns.min():.4f}%")
        print(f"Maksymalny dzienny zysk: {self.returns.max():.4f}%")

    def _test_heteroskedasticity_bp(self):
        """
        Test Breuscha-Pagana na heteroskedastyczność
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
            print(f"  Heteroskedastyczność: {'TAK' if bp_test[1] < 0.05 else 'NIE'}")

        except Exception as e:
            print(f"  Błąd podczas testu heteroskedastyczności: {e}")

    def visualize_data(self):
        """
        Tworzenie wykresów cen i stóp zwrotu dla PLN/EUR z CurrencyLayer
        """
        if self.raw_data is None or self.returns is None:
            print("✗ Brak danych do wizualizacji")
            return

        # Utworzenie subplotów w układzie 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analiza kursu {self.currency_pair} (CurrencyLayer API)',
                     fontsize=16, fontweight='bold')

        # Wykres 1: Kurs PLN/EUR w czasie
        axes[0, 0].plot(self.raw_data.index, self.raw_data['Close'],
                        color='blue', linewidth=1)
        axes[0, 0].set_title(f'Kurs {self.currency_pair} (1 PLN = ? EUR)')
        axes[0, 0].set_ylabel('Kurs EUR')
        axes[0, 0].grid(True, alpha=0.3)

        # Wykres 2: Stopy zwrotu w czasie
        axes[0, 1].plot(self.returns.index, self.returns,
                        color='red', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_title('Logarytmiczne stopy zwrotu (%)')
        axes[0, 1].set_ylabel('Stopa zwrotu (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Wykres 3: Histogram stóp zwrotu z krzywą normalną
        axes[1, 0].hist(self.returns, bins=30, density=True, alpha=0.7,
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
        Dopasowanie modelu GARCH(p,q) do danych PLN/EUR z CurrencyLayer
        """
        if self.returns is None:
            print("✗ Brak stóp zwrotu do modelowania")
            return False

        print(f"\n=== DOPASOWANIE MODELU GARCH({p},{q}) ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Źródło danych: CurrencyLayer API")
        print(f"Rozkład składnika losowego: {distribution}")

        try:
            # Definicja modelu GARCH
            self.model = arch_model(self.returns,
                                    mean='Zero',  # Zerowa średnia warunkowa
                                    vol='GARCH',  # Model GARCH dla wariancji
                                    p=p,  # Opóźnienia wariancji (GARCH)
                                    q=q,  # Opóźnienia kwadratów reszt (ARCH)
                                    dist=distribution)  # Rozkład składnika losowego

            # Dopasowanie modelu metodą maksymalnej wiarygodności
            self.results = self.model.fit(disp='off')

            # Wyświetlenie podsumowania wyników
            print(self.results.summary())

            # Analiza parametrów modelu
            self._analyze_model_parameters()
            return True

        except Exception as e:
            print(f"✗ Błąd podczas dopasowywania modelu GARCH: {e}")
            return False

    def _analyze_model_parameters(self):
        """
        Analiza i interpretacja oszacowanych parametrów modelu GARCH dla PLN/EUR
        """
        params = self.results.params

        print("\n" + "=" * 50)
        print(f"ANALIZA PARAMETRÓW MODELU GARCH(1,1) dla {self.currency_pair}")
        print(f"Źródło: CurrencyLayer API")
        print("=" * 50)

        # Wyciągnięcie parametrów
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
            annual_vol = unconditional_vol * np.sqrt(252)  # Roczna zmienność

            print(f"Długookresowa wariancja: {unconditional_var:.6f}")
            print(f"Długookresowa zmienność dzienna: {unconditional_vol:.4f}%")
            print(f"Długookresowa zmienność roczna: {annual_vol:.2f}%")
        else:
            print("⚠ Ostrzeżenie: Warunek stacjonarności niespełniony!")

        # Interpretacja parametrów w kontekście PLN/EUR
        print(f"\nInterpretacja parametrów dla kursu PLN/EUR:")
        print(f"- α = {alpha:.4f}: Siła reakcji na nowe szoki (np. decyzje NBP/EBC)")
        print(f"- β = {beta:.4f}: Trwałość zmienności")

        if alpha + beta < 1:
            half_life = np.log(0.5) / np.log(alpha + beta)
            print(f"- Połowa życia szoku zmienności: {half_life:.1f} dni")

        # Praktyczne wnioski
        print(f"\nPraktyczne wnioski:")
        if alpha > 0.1:
            print("- Rynek silnie reaguje na nowe informacje (wysokie α)")
        else:
            print("- Rynek słabo reaguje na nowe informacje (niskie α)")

        if beta > 0.8:
            print("- Zmienność jest bardzo trwała (wysokie β)")
        elif beta > 0.5:
            print("- Zmienność jest umiarkowanie trwała")
        else:
            print("- Zmienność szybko wygasa")

    def forecast_volatility(self, horizon=20):
        """
        Prognozowanie zmienności kursu PLN/EUR na zadany horyzont czasowy
        """
        if self.results is None:
            print("✗ Brak dopasowanego modelu do prognozowania")
            return pd.DataFrame()

        print(f"\n=== PROGNOZOWANIE ZMIENNOŚCI ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Horyzont: {horizon} dni")

        try:
            # Generowanie prognoz używając dopasowanego modelu
            forecasts = self.results.forecast(horizon=horizon)

            # Wyciągnięcie prognozowanych wariancji i przekształcenie na zmienność (%)
            forecast_variance = forecasts.variance.values[-1, :]
            forecast_volatility = np.sqrt(forecast_variance)

            # Utworzenie dat dla prognoz (dni robocze)
            last_date = self.returns.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=horizon, freq='B')

            # Dopasowanie długości
            min_length = min(len(forecast_dates), len(forecast_volatility))
            forecast_dates = forecast_dates[:min_length]
            forecast_volatility = forecast_volatility[:min_length]

            # Tworzenie DataFrame z prognozami
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast_volatility_daily': forecast_volatility,
                'forecast_volatility_annual': forecast_volatility * np.sqrt(252)
            })

            print("Pierwsze 10 prognoz zmienności:")
            print(forecast_df[['date', 'forecast_volatility_daily', 'forecast_volatility_annual']].head(10))

            print(f"\nŚrednia prognozowana zmienność:")
            print(f"Dzienna: {forecast_volatility.mean():.4f}%")
            print(f"Roczna: {forecast_volatility.mean() * np.sqrt(252):.2f}%")

            # Wizualizacja prognoz
            self._plot_volatility_forecast(forecast_df)

            return forecast_df

        except Exception as e:
            print(f"✗ Błąd podczas prognozowania: {e}")
            return pd.DataFrame()

    def _plot_volatility_forecast(self, forecast_df):
        """
        Wizualizacja historycznej i prognozowanej zmienności PLN/EUR
        """
        try:
            # Obliczenie historycznej zmienności (rolling std na oknie 20 dni)
            historical_vol = self.returns.rolling(window=20).std()

            # Utworzenie wykresu
            plt.figure(figsize=(15, 8))

            # Wykres historycznej zmienności
            plt.plot(historical_vol.index[-100:], historical_vol.iloc[-100:],
                     label='Historyczna zmienność (20-dniowa)', color='blue', linewidth=1.5)

            # Wykres prognozowanej zmienności
            if not forecast_df.empty:
                plt.plot(forecast_df['date'], forecast_df['forecast_volatility_daily'],
                         label='Prognoza GARCH', color='red', linewidth=2, linestyle='--')

            # Dodanie linii pionowej oznaczającej początek prognozy
            plt.axvline(x=self.returns.index[-1], color='green', linestyle=':',
                        label='Początek prognozy', alpha=0.7)

            plt.title(f'Historyczna i prognozowana zmienność kursu {self.currency_pair}\n(CurrencyLayer API)',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('Zmienność dzienna (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresu prognoz: {e}")

    def validate_model(self, test_size=0.2):
        """
        Walidacja modelu przez podział na zbiór treningowy i testowy
        """
        if self.returns is None:
            print("✗ Brak danych do walidacji")
            return {}

        print(f"\n=== WALIDACJA MODELU ===")
        print(f"Test size: {test_size}")

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

            # Prognozowanie na zbiorze testowym
            forecasts = train_results.forecast(horizon=len(test_returns))
            forecast_variance = forecasts.variance.values[-1, :]
            forecasted_volatilities = np.sqrt(forecast_variance)

            # Rzeczywista zmienność (absolutna wartość zwrotu jako proxy)
            actual_volatilities = np.abs(test_returns.values)

            # Dopasowanie długości
            min_length = min(len(forecasted_volatilities), len(actual_volatilities))
            forecasted_volatilities = forecasted_volatilities[:min_length]
            actual_volatilities = actual_volatilities[:min_length]

            # Obliczenie metryk błędów
            rmse = np.sqrt(mean_squared_error(actual_volatilities, forecasted_volatilities))
            mae = mean_absolute_error(actual_volatilities, forecasted_volatilities)

            # Bezpieczne obliczenie MAPE
            non_zero_actual = actual_volatilities[actual_volatilities != 0]
            non_zero_forecast = forecasted_volatilities[actual_volatilities != 0]

            if len(non_zero_actual) > 0:
                mape = np.mean(np.abs((non_zero_actual - non_zero_forecast) / non_zero_actual)) * 100
            else:
                mape = float('inf')

            print(f"\nMetryki jakości prognoz zmienności:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")

            # Interpretacja jakości
            if mape < 30:
                print("✓ Dobra jakość prognoz")
            elif mape < 60:
                print("○ Umiarkowana jakość prognoz")
            else:
                print("⚠ Słaba jakość prognoz")

            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'actual': actual_volatilities,
                'predicted': forecasted_volatilities
            }

        except Exception as e:
            print(f"✗ Błąd podczas walidacji modelu: {e}")
            return {}

    def run_complete_analysis(self):
        """
        Uruchomienie kompletnej analizy GARCH dla kursu PLN/EUR z CurrencyLayer
        """
        print("=" * 60)
        print(f"ANALIZA GARCH DLA KURSU {self.currency_pair}")
        print("ŹRÓDŁO: CURRENCYLAYER API")
        print("=" * 60)

        try:
            # Krok 1: Konfiguracja źródła danych
            if not self.setup_data_source():
                print("✗ Nie udało się skonfigurować źródła danych")
                return None

            # Krok 2: Obliczenie stóp zwrotu
            if not self.calculate_returns():
                print("✗ Nie udało się obliczyć stóp zwrotu")
                return None

            # Krok 3: Wizualizacja danych
            self.visualize_data()

            # Krok 4: Dopasowanie modelu GARCH
            if not self.fit_garch_model(p=1, q=1, distribution='normal'):
                print("✗ Nie udało się dopasować modelu GARCH")
                return None

            # Krok 5: Prognozowanie zmienności
            forecasts = self.forecast_volatility(horizon=20)

            # Krok 6: Walidacja modelu
            validation_results = self.validate_model(test_size=0.2)

            print("\n" + "=" * 60)
            print(f"ANALIZA {self.currency_pair} ZAKOŃCZONA POMYŚLNIE")
            print("=" * 60)

            return {
                'model_results': self.results,
                'forecasts': forecasts,
                'validation': validation_results
            }

        except Exception as e:
            print(f"✗ Błąd podczas analizy: {str(e)}")
            return None


# Główna część skryptu
if __name__ == "__main__":
    print("ANALIZA GARCH Z CURRENCYLAYER API")
    print("=" * 50)

    # Pobierz klucz API
    api_key = input("Podaj swój API key z currencylayer.com: ").strip()

    if not api_key:
        print("Nie podano API key!")
        print("Możesz też użyć istniejącego pliku CSV:")
        csv_file = input("Ścieżka do pliku CSV (lub Enter aby pominąć): ").strip()
        if not csv_file:
            print("Brak źródła danych. Kończę program.")
            exit()
    else:
        csv_file = None

    # Utworzenie instancji analizy GARCH
    garch_analyzer = GARCHAnalysisCurrencyLayer(
        api_key=api_key,
        base_currency='PLN',  # Złoty polski
        target_currency='EUR',  # Euro
        start_date='2025-05-01',  # Data początkowa (ogranicz żeby nie przekroczyć 100 API calls)
        end_date='2025-06-03',  # Data końcowa
        csv_file=csv_file  # Opcjonalny plik CSV
    )

    # Uruchomienie kompletnej analizy
    results = garch_analyzer.run_complete_analysis()

    # Sprawdzenie czy analiza się powiodła
    if results is not None:
        # Zapis wyników do pliku
        print("\nCzy chcesz zapisać wyniki do pliku CSV? (tak/nie)")
        try:
            response = input().lower().strip()

            if response in ['tak', 'yes', 'y', 't']:
                # Zapis prognoz do pliku CSV
                if not results['forecasts'].empty:
                    results['forecasts'].to_csv('currencylayer_garch_forecast.csv', index=False)
                    print("✓ Prognozy zapisane do pliku: currencylayer_garch_forecast.csv")

                # Zapis parametrów modelu do pliku tekstowego
                if results['model_results'] is not None:
                    with open('currencylayer_garch_summary.txt', 'w', encoding='utf-8') as f:
                        f.write("PODSUMOWANIE MODELU GARCH PLN/EUR (CurrencyLayer)\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(str(results['model_results'].summary()))

                    print("✓ Podsumowanie modelu zapisane do pliku: currencylayer_garch_summary.txt")
        except EOFError:
            print("Pomijam zapis plików")

    print("\nAnaliza kursu PLN/EUR z CurrencyLayer zakończona!")
