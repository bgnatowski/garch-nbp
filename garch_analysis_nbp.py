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

# Import naszego NBP fetcher'a
from nbp_fetcher import NBPFetcher

warnings.filterwarnings('ignore')  # Ukrycie ostrzeżeń dla czystości outputu

# Ustawienia dla wykresów - lepszy wygląd
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GARCHAnalysisNBP:
    """
    Klasa do kompleksowej analizy modelu GARCH dla kursów walutowych
    z wykorzystaniem danych z API NBP
    """

    def __init__(self, currency_code='eur', start_date='2023-01-01',
                 end_date='2024-12-31', table='a', csv_file=None):
        """
        Inicjalizacja klasy z podstawowymi parametrami
        """
        self.currency_code = currency_code.lower()  # Kod waluty (eur, usd, gbp, etc.)
        self.start_date = start_date  # Data początkowa pobierania danych
        self.end_date = end_date  # Data końcowa pobierania danych
        self.table = table.lower()  # Tabela NBP (a, b, c)
        self.csv_file = csv_file  # Opcjonalny plik CSV z danymi
        self.currency_pair = f"{currency_code.upper()}/PLN"  # Para walutowa
        self.raw_data = None  # Surowe dane cenowe
        self.returns = None  # Obliczone stopy zwrotu
        self.model = None  # Model GARCH
        self.results = None  # Wyniki dopasowania modelu
        self.fetcher = None  # Obiekt fetcher'a NBP

    def setup_data_source(self):
        """
        Konfiguracja źródła danych - API NBP lub plik CSV
        """
        print("=== KONFIGURACJA ŹRÓDŁA DANYCH ===")

        # Jeśli podano plik CSV, próbujemy z niego
        if self.csv_file:
            print(f"Próba wczytania danych z pliku CSV: {self.csv_file}")
            try:
                self.raw_data = pd.read_csv(self.csv_file, parse_dates=True, index_col=0)

                # Sprawdzenie czy plik ma odpowiednie kolumny
                if 'Close' not in self.raw_data.columns:
                    # Próba użycia innych nazw kolumn
                    if 'mid' in self.raw_data.columns:
                        self.raw_data['Close'] = self.raw_data['mid']
                    elif 'NBP_Rate' in self.raw_data.columns:
                        self.raw_data['Close'] = self.raw_data['NBP_Rate']
                    else:
                        # Użyj pierwszej numerycznej kolumny
                        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            self.raw_data['Close'] = self.raw_data[numeric_cols[0]]
                        else:
                            raise ValueError("Brak odpowiednich kolumn numerycznych w pliku CSV")

                # Dodanie brakujących kolumn
                if 'Open' not in self.raw_data.columns:
                    self.raw_data['Open'] = self.raw_data['Close']
                if 'High' not in self.raw_data.columns:
                    self.raw_data['High'] = self.raw_data['Close'] * 1.001
                if 'Low' not in self.raw_data.columns:
                    self.raw_data['Low'] = self.raw_data['Close'] * 0.999
                if 'Volume' not in self.raw_data.columns:
                    self.raw_data['Volume'] = 1000000

                print(f"✓ Wczytano {len(self.raw_data)} obserwacji z pliku CSV")
                return True

            except Exception as e:
                print(f"✗ Błąd wczytywania CSV: {e}")
                print("Próba pobrania danych z API NBP...")

        # Pobieranie danych z API NBP
        print("Inicjalizacja połączenia z API NBP...")
        self.fetcher = NBPFetcher()

        # Test połączenia z API
        if not self.fetcher.test_api_connection():
            print("✗ Nie można nawiązać połączenia z API NBP")
            return False

        # Pobieranie danych historycznych
        print(f"\nPobieranie danych historycznych dla {self.currency_code.upper()}...")
        if self.fetcher.fetch_historical_data(
                self.currency_code, self.start_date, self.end_date, self.table
        ):
            self.raw_data = self.fetcher.get_data_for_garch()
            print("✓ Dane pobrane z API NBP")

            # Automatyczny zapis jako backup
            backup_filename = f"nbp_{self.currency_code}_{self.start_date}_{self.end_date}.csv"
            self.fetcher.save_data(backup_filename)

            return True
        else:
            print("✗ Nie udało się pobrać danych z API NBP")
            return False

    def calculate_returns(self):
        """
        Obliczanie logarytmicznych stóp zwrotu z kursów NBP
        """
        if self.raw_data is None:
            print("✗ Brak danych do obliczenia stóp zwrotu")
            return False

        # Logarytmiczne stopy zwrotu: log(P_t / P_{t-1}) * 100 (w procentach)
        self.returns = 100 * np.log(self.raw_data['Close'] / self.raw_data['Close'].shift(1))

        # Usunięcie pierwszej obserwacji (NaN) powstałej przez przesunięcie
        self.returns = self.returns.dropna()

        print(f"\n✓ Obliczono {len(self.returns)} stóp zwrotu dla kursu {self.currency_pair}")
        print("Podstawowe statystyki stóp zwrotu:")
        print(self.returns.describe())

        # Sprawdzenie własności statystycznych zwrotów
        self._analyze_returns_properties()
        return True

    def _analyze_returns_properties(self):
        """
        Analiza właściwości statystycznych stóp zwrotu
        """
        print("\n" + "=" * 60)
        print(f"ANALIZA WŁAŚCIWOŚCI STÓP ZWROTU {self.currency_pair}")
        print(f"Źródło: API NBP (tabela {self.table.upper()})")
        print("=" * 60)

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

        # Dodatkowe statystyki specyficzne dla kursów NBP
        print(f"\nDodatkowe statystyki dla {self.currency_pair} (NBP):")
        print(f"Średnia dzienna zmiana: {self.returns.mean():.4f}%")
        print(f"Dzienna zmienność: {self.returns.std():.4f}%")
        print(f"Roczna zmienność: {self.returns.std() * np.sqrt(252):.2f}%")
        print(f"Maksymalna dzienna strata: {self.returns.min():.4f}%")
        print(f"Maksymalny dzienny zysk: {self.returns.max():.4f}%")
        print(f"Procent dni ze wzrostem: {(self.returns > 0).mean() * 100:.1f}%")
        print(f"Procent dni ze spadkiem: {(self.returns < 0).mean() * 100:.1f}%")

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
        Tworzenie wykresów kursów i stóp zwrotu z danych NBP
        """
        if self.raw_data is None or self.returns is None:
            print("✗ Brak danych do wizualizacji")
            return

        # Utworzenie subplotów w układzie 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analiza kursu {self.currency_pair} (NBP API)',
                     fontsize=16, fontweight='bold')

        # Wykres 1: Kurs w czasie
        axes[0, 0].plot(self.raw_data.index, self.raw_data['Close'],
                        color='blue', linewidth=1)
        axes[0, 0].set_title(f'Kurs {self.currency_pair} (PLN za 1 {self.currency_code.upper()})')
        axes[0, 0].set_ylabel('Kurs (PLN)')
        axes[0, 0].grid(True, alpha=0.3)

        # Wykres 2: Stopy zwrotu w czasie
        axes[0, 1].plot(self.returns.index, self.returns,
                        color='red', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_title('Logarytmiczne stopy zwrotu (%)')
        axes[0, 1].set_ylabel('Stopa zwrotu (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Wykres 3: Histogram stóp zwrotu z krzywą normalną
        n_bins = min(50, len(self.returns) // 5)  # Dostosowanie liczby bins do rozmiaru próby
        axes[1, 0].hist(self.returns, bins=n_bins, density=True, alpha=0.7,
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
        Dopasowanie modelu GARCH(p,q) do danych z NBP
        """
        if self.returns is None:
            print("✗ Brak stóp zwrotu do modelowania")
            return False

        print(f"\n=== DOPASOWANIE MODELU GARCH({p},{q}) ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Źródło danych: NBP API (tabela {self.table.upper()})")
        print(f"Liczba obserwacji: {len(self.returns)}")
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
        Analiza i interpretacja oszacowanych parametrów modelu GARCH
        """
        params = self.results.params

        print("\n" + "=" * 60)
        print(f"ANALIZA PARAMETRÓW MODELU GARCH(1,1) dla {self.currency_pair}")
        print(f"Źródło: NBP API")
        print("=" * 60)

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

        # Interpretacja parametrów w kontekście polskiej waluty
        print(f"\nInterpretacja parametrów dla kursu {self.currency_pair}:")
        print(f"- α = {alpha:.4f}: Siła reakcji na nowe szoki (np. decyzje NBP, dane makro)")
        print(f"- β = {beta:.4f}: Trwałość zmienności (pamięć rynku)")

        if alpha + beta < 1:
            half_life = np.log(0.5) / np.log(alpha + beta)
            print(f"- Połowa życia szoku zmienności: {half_life:.1f} dni")

        # Praktyczne wnioski
        print(f"\nPraktyczne wnioski:")
        if alpha > 0.15:
            print("- Rynek silnie reaguje na nowe informacje (wysokie α)")
        elif alpha > 0.05:
            print("- Rynek umiarkowanie reaguje na nowe informacje")
        else:
            print("- Rynek słabo reaguje na nowe informacje (niskie α)")

        if beta > 0.85:
            print("- Zmienność jest bardzo trwała (wysokie β)")
        elif beta > 0.5:
            print("- Zmienność jest umiarkowanie trwała")
        else:
            print("- Zmienność szybko wygasa")

        # Ocena jakości modelu
        aic = self.results.aic
        bic = self.results.bic
        log_likelihood = self.results.loglikelihood

        print(f"\nJakość dopasowania modelu:")
        print(f"- Log-likelihood: {log_likelihood:.2f}")
        print(f"- AIC: {aic:.2f}")
        print(f"- BIC: {bic:.2f}")

    def forecast_volatility(self, horizon=30):
        """
        Prognozowanie zmienności kursu na zadany horyzont czasowy
        """
        if self.results is None:
            print("✗ Brak dopasowanego modelu do prognozowania")
            return pd.DataFrame()

        print(f"\n=== PROGNOZOWANIE ZMIENNOŚCI ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Horyzont: {horizon} dni roboczych")

        try:
            # Generowanie prognoz używając dopasowanego modelu
            forecasts = self.results.forecast(horizon=horizon)

            # Wyciągnięcie prognozowanych wariancji i przekształcenie na zmienność (%)
            forecast_variance = forecasts.variance.values[-1, :]
            forecast_volatility = np.sqrt(forecast_variance)

            # Utworzenie dat dla prognoz (dni robocze - kiedy NBP publikuje kursy)
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

            print(f"\nStatystyki prognoz:")
            print(f"Średnia prognozowana zmienność dzienna: {forecast_volatility.mean():.4f}%")
            print(f"Średnia prognozowana zmienność roczna: {forecast_volatility.mean() * np.sqrt(252):.2f}%")
            print(f"Zakres prognoz dziennych: {forecast_volatility.min():.4f}% - {forecast_volatility.max():.4f}%")

            # Wizualizacja prognoz
            self._plot_volatility_forecast(forecast_df)

            return forecast_df

        except Exception as e:
            print(f"✗ Błąd podczas prognozowania: {e}")
            return pd.DataFrame()

    def _plot_volatility_forecast(self, forecast_df):
        """
        Wizualizacja historycznej i prognozowanej zmienności
        """
        try:
            # Obliczenie historycznej zmienności (rolling std na oknie 30 dni)
            historical_vol = self.returns.rolling(window=30).std()

            # Utworzenie wykresu
            plt.figure(figsize=(15, 8))

            # Wykres historycznej zmienności (ostatnie obserwacje dla czytelności)
            n_hist = min(200, len(historical_vol))
            plt.plot(historical_vol.index[-n_hist:], historical_vol.iloc[-n_hist:],
                     label='Historyczna zmienność (30-dniowa)', color='blue', linewidth=1.5)

            # Wykres prognozowanej zmienności
            if not forecast_df.empty:
                plt.plot(forecast_df['date'], forecast_df['forecast_volatility_daily'],
                         label='Prognoza GARCH', color='red', linewidth=2, linestyle='--')

            # Dodanie linii pionowej oznaczającej początek prognozy
            plt.axvline(x=self.returns.index[-1], color='green', linestyle=':',
                        label='Początek prognozy', alpha=0.7)

            plt.title(f'Historyczna i prognozowana zmienność kursu {self.currency_pair}\n(NBP API)',
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

            if len(test_returns) < 5:
                print("⚠ Zbyt mały zbiór testowy - wyniki mogą być niewiarygodne")

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
            non_zero_actual = actual_volatilities[actual_volatilities > 1e-10]
            non_zero_forecast = forecasted_volatilities[actual_volatilities > 1e-10]

            if len(non_zero_actual) > 0:
                mape = np.mean(np.abs((non_zero_actual - non_zero_forecast) / non_zero_actual)) * 100
            else:
                mape = float('inf')

            # Korelacja między prognozami a rzeczywistością
            correlation = np.corrcoef(actual_volatilities, forecasted_volatilities)[0, 1]

            print(f"\nMetryki jakości prognoz zmienności:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"Korelacja: {correlation:.4f}")

            # Interpretacja jakości
            if mape < 25:
                quality = "✓ Bardzo dobra jakość prognoz"
            elif mape < 50:
                quality = "✓ Dobra jakość prognoz"
            elif mape < 75:
                quality = "○ Umiarkowana jakość prognoz"
            else:
                quality = "⚠ Słaba jakość prognoz"

            print(quality)

            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'correlation': correlation,
                'actual': actual_volatilities,
                'predicted': forecasted_volatilities
            }

        except Exception as e:
            print(f"✗ Błąd podczas walidacji modelu: {e}")
            return {}

    def run_complete_analysis(self):
        """
        Uruchomienie kompletnej analizy GARCH z danymi NBP
        """
        print("=" * 70)
        print(f"ANALIZA GARCH DLA KURSU {self.currency_pair}")
        print("ŹRÓDŁO: NARODOWY BANK POLSKI (NBP) API")
        print("=" * 70)

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
            forecasts = self.forecast_volatility(horizon=30)

            # Krok 6: Walidacja modelu
            validation_results = self.validate_model(test_size=0.2)

            print("\n" + "=" * 70)
            print(f"ANALIZA {self.currency_pair} ZAKOŃCZONA POMYŚLNIE")
            print("=" * 70)

            return {
                'model_results': self.results,
                'forecasts': forecasts,
                'validation': validation_results,
                'raw_data': self.raw_data,
                'returns': self.returns
            }

        except Exception as e:
            print(f"✗ Błąd podczas analizy: {str(e)}")
            return None


# Główna część skryptu
if __name__ == "__main__":
    print("ANALIZA GARCH Z DANYMI NBP")
    print("=" * 50)

    # Konfiguracja analizy
    print(
        "Dostępne waluty w tabeli A NBP: EUR, USD, GBP, CHF, JPY, CZK, SEK, NOK, DKK, CAD, AUD, HUF, RON, BGN, TRY, ILS, CLP, PHP, MXN, ZAR, BRL, MYR, RUB, IDR, INR, KRW, CNY, XDR, THB, SGD, HKD, UAH, ISK, HRK, SKK")

    # Pobierz parametry od użytkownika
    currency = input("Kod waluty [eur]: ").strip().lower()
    if not currency:
        currency = "eur"

    start_date = input("Data początkowa (YYYY-MM-DD) [2022-01-01]: ").strip()
    if not start_date:
        start_date = "2022-01-01"

    end_date = input("Data końcowa (YYYY-MM-DD) [2024-12-31]: ").strip()
    if not end_date:
        end_date = "2024-12-31"

    csv_file = input("Ścieżka do pliku CSV (lub Enter dla API): ").strip()
    if not csv_file:
        csv_file = None

    # Utworzenie instancji analizy GARCH
    garch_analyzer = GARCHAnalysisNBP(
        currency_code=currency,  # Kod waluty
        start_date=start_date,  # Data początkowa
        end_date=end_date,  # Data końcowa
        table='a',  # Tabela NBP
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
                    forecast_filename = f"nbp_garch_forecast_{currency}_{start_date}_{end_date}.csv"
                    results['forecasts'].to_csv(forecast_filename, index=False)
                    print(f"✓ Prognozy zapisane do pliku: {forecast_filename}")

                # Zapis parametrów modelu do pliku tekstowego
                if results['model_results'] is not None:
                    summary_filename = f"nbp_garch_summary_{currency}_{start_date}_{end_date}.txt"
                    with open(summary_filename, 'w', encoding='utf-8') as f:
                        f.write(f"PODSUMOWANIE MODELU GARCH {currency.upper()}/PLN (NBP)\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(f"Okres analizy: {start_date} - {end_date}\n")
                        f.write(f"Liczba obserwacji: {len(results['returns'])}\n\n")
                        f.write(str(results['model_results'].summary()))

                    print(f"✓ Podsumowanie modelu zapisane do pliku: {summary_filename}")
        except EOFError:
            print("Pomijam zapis plików")

    print(f"\nAnaliza kursu {currency.upper()}/PLN z NBP API zakończona!")
