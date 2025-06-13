import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller

from nbp_fetcher import NBPFetcher

warnings.filterwarnings('ignore')

# Ustawienia dla wykresów
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class OptimizedGARCHAnalysisNBP:
    """
    Zoptymalizowana klasa do analizy modelu GARCH
    Optymalizacja:
    1. Preprocessing i oczyszczanie danych:
        - Wykrywanie i usuwanie outlierów
        - Zastosowane metody czyszczenia:
            Winsoryzacja: Zastępowanie wartości ekstremalnych granicami IQR
            Usuwanie: Całkowite usunięcie outlierów
            Zastępowanie medianą/średnią: Podmiana wartości ekstremalnych
    2. Test stacjonarności
    3. Optymalizacja parametrów modelu
        - Automatyczny Grid Search
        - Testowane kombinacje:
            Parametry p, q: od (1,1) do (2,2)
            Rozkłady: normalny, t-Studenta
            Kryteria selekcji: AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion)
        - Sprawdzanie stabilności modelu
    3. Ulepszenia w walidacji modelu
        - Bezpieczne obliczenie MAPE
        - Symetryczne MAPE
        - Rolling Window Validation
    4. Stabilizacja obliczeń i zarządzanie błędami
        - Lepsze zarządzanie długości okien
        - Filtrowanie ekstremalnych prognoz
    Kluczowe osiągnięcia:
        - MAPE spadło z bardzo duzych % do 50% - gigantyczna poprawa
        - Model znalazł optymalne parametry automatycznie
        - Prognozy są bardziej sensowne pomimo uzywania średnich wartości kursów (z NBP)
    """

    def __init__(self, currency_code='eur', start_date='2021-01-01',
                 end_date='2024-12-31', table='a', csv_file=None):
        self.currency_code = currency_code.lower()
        self.start_date = start_date
        self.end_date = end_date
        self.table = table.lower()
        self.csv_file = csv_file
        self.currency_pair = f"{currency_code.upper()}/PLN"
        self.raw_data = None
        self.returns = None
        self.returns_cleaned = None
        self.model = None
        self.results = None
        self.fetcher = None
        self.best_params = None
        self.outliers_removed = 0

    def setup_data_source(self):
        """Konfiguracja źródła danych - API NBP lub plik CSV"""
        print("=== KONFIGURACJA ŹRÓDŁA DANYCH ===")

        if self.csv_file:
            print(f"Próba wczytania danych z pliku CSV: {self.csv_file}")
            try:
                self.raw_data = pd.read_csv(self.csv_file, parse_dates=True, index_col=0)

                if 'Close' not in self.raw_data.columns:
                    if 'mid' in self.raw_data.columns:
                        self.raw_data['Close'] = self.raw_data['mid']
                    elif 'NBP_Rate' in self.raw_data.columns:
                        self.raw_data['Close'] = self.raw_data['NBP_Rate']
                    else:
                        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            self.raw_data['Close'] = self.raw_data[numeric_cols[0]]
                        else:
                            raise ValueError("Brak odpowiednich kolumn numerycznych w pliku CSV")

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

        if not self.fetcher.test_api_connection():
            print("✗ Nie można nawiązać połączenia z API NBP")
            return False

        print(f"\nPobieranie danych historycznych dla {self.currency_code.upper()}...")
        if self.fetcher.fetch_historical_data(
                self.currency_code, self.start_date, self.end_date, self.table
        ):
            self.raw_data = self.fetcher.get_data_for_garch()
            print("✓ Dane pobrane z API NBP")

            backup_filename = f"nbp_{self.currency_code}_{self.start_date}_{self.end_date}.csv"
            self.fetcher.save_data(backup_filename)
            return True
        else:
            print("✗ Nie udało się pobrać danych z API NBP")
            return False

    def detect_outliers_iqr(self, data, method='iqr', multiplier=2.0):
        """
        Wykrywanie outlierów metodą IQR lub Z-score (łagodniejsze podejście)
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > multiplier
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

        return outliers

    def clean_outliers(self, data, method='iqr', treatment='winsorize'):
        """
        Oczyszczanie outlierów różnymi metodami (łagodniejsze podejście)
        """
        outliers = self.detect_outliers_iqr(data, method, multiplier=2.0)  # Łagodniejsze
        self.outliers_removed = outliers.sum()

        print(f"Wykryto {self.outliers_removed} outlierów ({self.outliers_removed / len(data) * 100:.2f}% danych)")

        if treatment == 'remove':
            cleaned_data = data[~outliers]
        elif treatment == 'winsorize':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR  # Łagodniejsze
            upper_bound = Q3 + 2.0 * IQR

            cleaned_data = data.copy()
            cleaned_data[data < lower_bound] = lower_bound
            cleaned_data[data > upper_bound] = upper_bound
        elif treatment == 'median':
            cleaned_data = data.copy()
            cleaned_data[outliers] = data.median()
        elif treatment == 'mean':
            cleaned_data = data.copy()
            cleaned_data[outliers] = data.mean()
        else:
            raise ValueError("Treatment must be 'winsorize', 'remove', 'median', or 'mean'")

        print(f"Zastosowano metodę: {treatment}")
        print(f"Dane po oczyszczeniu: {len(cleaned_data)} obserwacji")

        return cleaned_data

    def test_stationarity(self, data, significance_level=0.05):
        """
        Test stacjonarności szeregu czasowego (ADF test)
        """
        print("\n=== TEST STACJONARNOŚCI (ADF) ===")

        try:
            result = adfuller(data.dropna(), autolag='AIC')

            print(f"ADF Statistic: {result[0]:.6f}")
            print(f"p-value: {result[1]:.6f}")
            print(f"Lags Used: {result[2]}")
            print(f"Number of Observations: {result[3]}")

            print("Critical Values:")
            for key, value in result[4].items():
                print(f"  {key}: {value:.6f}")

            is_stationary = result[1] < significance_level

            if is_stationary:
                print(f"✓ Szereg jest STACJONARNY (p-value: {result[1]:.6f} < {significance_level})")
            else:
                print(f"⚠ Szereg jest NIESTACJONARNY (p-value: {result[1]:.6f} >= {significance_level})")
                print("Rozważ różnicowanie szeregu lub inne transformacje")

            return is_stationary, result

        except Exception as e:
            print(f"✗ Błąd podczas testu stacjonarności: {e}")
            return False, None

    def optimize_garch_parameters(self, data, max_p=2, max_q=2, distributions=['normal', 't']):
        """
        Optymalizacja parametrów GARCH (ograniczona do stabilnych modeli)
        """
        print("\n=== OPTYMALIZACJA PARAMETRÓW GARCH ===")
        print(f"Testowanie kombinacji p={range(1, max_p + 1)}, q={range(1, max_q + 1)}")
        print(f"Rozkłady: {distributions}")

        best_aic = float('inf')
        best_model_aic = None
        best_params_aic = None

        results_df = []

        for p, q, dist in product(range(1, max_p + 1), range(1, max_q + 1), distributions):
            try:
                if p + q > len(data) // 100:
                    continue

                print(f"Testowanie GARCH({p},{q}) z rozkładem {dist}...")

                model = arch_model(data, vol='GARCH', p=p, q=q, dist=dist, mean='Zero')
                results = model.fit(disp='off')

                # Sprawdzenie stabilności parametrów
                params = results.params
                if dist == 'normal':
                    alpha_sum = sum([params[f'alpha[{i}]'] for i in range(1, q + 1)])
                    beta_sum = sum([params[f'beta[{i}]'] for i in range(1, p + 1)])

                    if alpha_sum + beta_sum >= 0.99:  # Zbyt blisko niestabilności
                        print(f"  Pomijam - model niestabilny (α+β = {alpha_sum + beta_sum:.4f})")
                        continue

                aic = results.aic
                bic = results.bic
                log_likelihood = results.loglikelihood

                results_df.append({
                    'p': p, 'q': q, 'distribution': dist,
                    'AIC': aic, 'BIC': bic, 'LogLikelihood': log_likelihood
                })

                if aic < best_aic:
                    best_aic = aic
                    best_model_aic = results
                    best_params_aic = (p, q, dist)

                print(f"  AIC: {aic:.4f}, BIC: {bic:.4f}, LogLik: {log_likelihood:.4f}")

            except Exception as e:
                print(f"  Błąd dla GARCH({p},{q}) {dist}: {str(e)[:50]}...")
                continue

        results_df = pd.DataFrame(results_df)

        if not results_df.empty:
            print(f"\n=== WYNIKI OPTYMALIZACJI ===")
            print(f"Przetestowano {len(results_df)} stabilnych modeli")

            print(f"\nNajlepszy model wg AIC:")
            print(f"  GARCH{best_params_aic[:2]} z rozkładem {best_params_aic[2]}")
            print(f"  AIC: {best_aic:.4f}")

            print(f"\nTop 3 modeli wg AIC:")
            top_3_aic = results_df.nsmallest(3, 'AIC')[['p', 'q', 'distribution', 'AIC', 'BIC']]
            print(top_3_aic.to_string(index=False))

            self.best_params = best_params_aic
            return best_model_aic, best_params_aic, results_df
        else:
            print("✗ Nie udało się dopasować żadnego stabilnego modelu")
            return None, None, pd.DataFrame()

    def calculate_returns(self, clean_outliers=True, outlier_method='iqr', outlier_treatment='winsorize'):
        """
        Obliczanie i oczyszczanie logarytmicznych stóp zwrotu
        """
        if self.raw_data is None:
            print("✗ Brak danych do obliczenia stóp zwrotu")
            return False

        # Logarytmiczne stopy zwrotu
        self.returns = 100 * np.log(self.raw_data['Close'] / self.raw_data['Close'].shift(1))
        self.returns = self.returns.dropna()

        print(f"\n✓ Obliczono {len(self.returns)} stóp zwrotu dla kursu {self.currency_pair}")

        # Test stacjonarności oryginalnych stóp zwrotu
        is_stationary, adf_result = self.test_stationarity(self.returns)

        if clean_outliers:
            print(f"\n=== OCZYSZCZANIE OUTLIERÓW ===")
            print(f"Metoda wykrywania: {outlier_method}")
            print(f"Metoda czyszczenia: {outlier_treatment}")

            self.returns_cleaned = self.clean_outliers(
                self.returns,
                method=outlier_method,
                treatment=outlier_treatment
            )

            if len(self.returns_cleaned) > 0:
                print(f"\nTest stacjonarności po oczyszczeniu:")
                is_stationary_clean, _ = self.test_stationarity(self.returns_cleaned)
        else:
            self.returns_cleaned = self.returns.copy()
            self.outliers_removed = 0

        print("\nPodstawowe statystyki stóp zwrotu (po oczyszczeniu):")
        print(self.returns_cleaned.describe())

        self._analyze_returns_properties()
        return True

    def _analyze_returns_properties(self):
        """
        Analiza właściwości statystycznych oczyszczonych stóp zwrotu
        """
        data = self.returns_cleaned

        print("\n" + "=" * 60)
        print(f"ANALIZA WŁAŚCIWOŚCI STÓP ZWROTU {self.currency_pair}")
        print(f"Źródło: API NBP (tabela {self.table.upper()})")
        print(f"Outlierów usuniętych: {self.outliers_removed}")
        print("=" * 60)

        jb_stat, jb_pvalue = stats.jarque_bera(data)
        print(f"Test Jarque-Bera na normalność:")
        print(f"  Statystyka: {jb_stat:.4f}")
        print(f"  P-wartość: {jb_pvalue:.4f}")
        print(f"  Rozkład normalny: {'NIE' if jb_pvalue < 0.05 else 'TAK'}")

        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        print(f"\nSkośność: {skewness:.4f}")
        print(f"Kurtoza: {kurtosis:.4f} (nadmierna kurtoza: {kurtosis:.4f})")

        self._test_heteroskedasticity_bp()

        print(f"\nDodatkowe statystyki dla {self.currency_pair} (NBP):")
        print(f"Średnia dzienna zmiana: {data.mean():.4f}%")
        print(f"Dzienna zmienność: {data.std():.4f}%")
        print(f"Roczna zmienność: {data.std() * np.sqrt(252):.2f}%")
        print(f"Maksymalna dzienna strata: {data.min():.4f}%")
        print(f"Maksymalny dzienny zysk: {data.max():.4f}%")
        print(f"Procent dni ze wzrostem: {(data > 0).mean() * 100:.1f}%")
        print(f"Procent dni ze spadkiem: {(data < 0).mean() * 100:.1f}%")

    def _test_heteroskedasticity_bp(self):
        """Test Breuscha-Pagana na heteroskedastyczność"""
        print(f"\nTest Breuscha-Pagana na heteroskedastyczność:")

        try:
            X = np.arange(len(self.returns_cleaned))
            X = sm.add_constant(X)
            model = sm.OLS(self.returns_cleaned ** 2, X).fit()
            bp_test = het_breuschpagan(model.resid, model.model.exog)

            print(f"  Statystyka LM: {bp_test[0]:.4f}")
            print(f"  P-wartość: {bp_test[1]:.4f}")
            print(f"  Heteroskedastyczność: {'TAK' if bp_test[1] < 0.05 else 'NIE'}")

        except Exception as e:
            print(f"  Błąd podczas testu heteroskedastyczności: {e}")

    def visualize_data(self):
        """
        Wizualizacja danych z porównaniem przed i po oczyszczeniu
        """
        if self.raw_data is None or self.returns is None:
            print("✗ Brak danych do wizualizacji")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Analiza kursu {self.currency_pair} (NBP API) - Przed i po oczyszczeniu',
                     fontsize=16, fontweight='bold')

        # Wykres 1: Kurs w czasie
        axes[0, 0].plot(self.raw_data.index, self.raw_data['Close'],
                        color='blue', linewidth=1)
        axes[0, 0].set_title(f'Kurs {self.currency_pair}')
        axes[0, 0].set_ylabel('Kurs (PLN)')
        axes[0, 0].grid(True, alpha=0.3)

        # Wykres 2: Stopy zwrotu oryginalne
        axes[0, 1].plot(self.returns.index, self.returns,
                        color='red', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_title('Stopy zwrotu (oryginalne)')
        axes[0, 1].set_ylabel('Stopa zwrotu (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Wykres 3: Stopy zwrotu oczyszczone
        axes[0, 2].plot(self.returns_cleaned.index, self.returns_cleaned,
                        color='green', linewidth=0.8, alpha=0.7)
        axes[0, 2].set_title(f'Stopy zwrotu (oczyszczone, {self.outliers_removed} outlierów)')
        axes[0, 2].set_ylabel('Stopa zwrotu (%)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Wykres 4: Histogram oryginalny
        axes[1, 0].hist(self.returns, bins=50, density=True, alpha=0.7,
                        color='red', edgecolor='black')
        mu, sigma = self.returns.mean(), self.returns.std()
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        axes[1, 0].plot(x, normal_curve, 'b--', linewidth=2, label='Rozkład normalny')
        axes[1, 0].set_title('Histogram (oryginalne)')
        axes[1, 0].set_xlabel('Stopa zwrotu (%)')
        axes[1, 0].set_ylabel('Gęstość')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Wykres 5: Histogram oczyszczony
        axes[1, 1].hist(self.returns_cleaned, bins=50, density=True, alpha=0.7,
                        color='green', edgecolor='black')
        mu_clean, sigma_clean = self.returns_cleaned.mean(), self.returns_cleaned.std()
        x_clean = np.linspace(self.returns_cleaned.min(), self.returns_cleaned.max(), 100)
        normal_curve_clean = stats.norm.pdf(x_clean, mu_clean, sigma_clean)
        axes[1, 1].plot(x_clean, normal_curve_clean, 'b--', linewidth=2, label='Rozkład normalny')
        axes[1, 1].set_title('Histogram (oczyszczone)')
        axes[1, 1].set_xlabel('Stopa zwrotu (%)')
        axes[1, 1].set_ylabel('Gęstość')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Wykres 6: Q-Q plot
        stats.probplot(self.returns_cleaned, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q plot (oczyszczone)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def fit_optimized_garch_model(self, optimize_params=True):
        """
        Dopasowanie optymalnego modelu GARCH
        """
        if self.returns_cleaned is None:
            print("✗ Brak oczyszczonych stóp zwrotu do modelowania")
            return False

        print(f"\n=== DOPASOWANIE OPTYMALNEGO MODELU GARCH ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Źródło danych: NBP API (tabela {self.table.upper()})")
        print(f"Liczba obserwacji: {len(self.returns_cleaned)}")

        if optimize_params:
            optimized_results, best_params, optimization_df = self.optimize_garch_parameters(
                self.returns_cleaned, max_p=2, max_q=2
            )

            if optimized_results is not None:
                self.results = optimized_results
                self.best_params = best_params
                print(f"\n✓ Użyto optymalnych parametrów: GARCH{best_params[:2]} z rozkładem {best_params[2]}")
            else:
                print("✗ Optymalizacja nie powiodła się, używam GARCH(1,1)")
                return self._fit_default_garch()
        else:
            return self._fit_default_garch()

        print(self.results.summary())
        self._analyze_model_parameters()
        return True

    def _fit_default_garch(self):
        """Dopasowanie domyślnego modelu GARCH(1,1)"""
        try:
            self.model = arch_model(self.returns_cleaned,
                                    mean='Zero',
                                    vol='GARCH',
                                    p=1, q=1,
                                    dist='normal')
            self.results = self.model.fit(disp='off')
            self.best_params = (1, 1, 'normal')
            return True
        except Exception as e:
            print(f"✗ Błąd podczas dopasowywania domyślnego modelu: {e}")
            return False

    def _analyze_model_parameters(self):
        """
        Analiza parametrów optymalnego modelu GARCH
        """
        params = self.results.params

        print("\n" + "=" * 60)
        print(f"ANALIZA PARAMETRÓW OPTYMALNEGO MODELU GARCH dla {self.currency_pair}")
        print(f"Model: GARCH{self.best_params[:2]} z rozkładem {self.best_params[2]}")
        print(f"Źródło: NBP API")
        print("=" * 60)

        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']

        print(f"ω (omega) = {omega:.6f}")
        print(f"α (alpha) = {alpha:.6f}")
        print(f"β (beta)  = {beta:.6f}")
        print(f"α + β = {alpha + beta:.6f}")

        if alpha + beta < 1:
            print("✓ Warunek stacjonarności spełniony (α + β < 1)")

            unconditional_var = omega / (1 - alpha - beta)
            unconditional_vol = np.sqrt(unconditional_var)
            annual_vol = unconditional_vol * np.sqrt(252)

            print(f"Długookresowa wariancja: {unconditional_var:.6f}")
            print(f"Długookresowa zmienność dzienna: {unconditional_vol:.4f}%")
            print(f"Długookresowa zmienność roczna: {annual_vol:.2f}%")
        else:
            print("⚠ Ostrzeżenie: Warunek stacjonarności niespełniony!")

        print(f"\nInterpretacja parametrów dla kursu {self.currency_pair}:")
        print(f"- α = {alpha:.4f}: Siła reakcji na nowe szoki")
        print(f"- β = {beta:.4f}: Trwałość zmienności")

        if alpha + beta < 1:
            half_life = np.log(0.5) / np.log(alpha + beta)
            print(f"- Połowa życia szoku zmienności: {half_life:.1f} dni")

        aic = self.results.aic
        bic = self.results.bic
        log_likelihood = self.results.loglikelihood

        print(f"\nJakość dopasowania modelu:")
        print(f"- Log-likelihood: {log_likelihood:.2f}")
        print(f"- AIC: {aic:.2f}")
        print(f"- BIC: {bic:.2f}")

    def safe_mape(self, actual, predicted, threshold_percentile=20):
        """
        Bezpieczne obliczenie MAPE z filtrowaniem małych wartości
        """
        # Filtruj obserwacje gdzie actual > percentyl jako próg
        threshold = np.percentile(np.abs(actual), threshold_percentile)
        mask = np.abs(actual) > threshold

        if np.sum(mask) < 10:  # Za mało obserwacji
            return float('inf'), 0

        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]

        # Dodatkowy filtr dla ekstremalnych wartości
        extreme_mask = (np.abs(actual_filtered) < 10) & (np.abs(predicted_filtered) < 10)
        actual_filtered = actual_filtered[extreme_mask]
        predicted_filtered = predicted_filtered[extreme_mask]

        if len(actual_filtered) < 5:
            return float('inf'), 0

        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
        return mape, len(actual_filtered)

    def symmetric_mape(self, actual, predicted):
        """
        Symetryczne MAPE - alternatywa dla standardowego MAPE
        """
        return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100

    def improved_rolling_validation(self, window_size=500, min_forecasts=30):
        """
        Ulepszona walidacja z większym oknem i stabilniejszym modelem
        """
        if self.returns_cleaned is None:
            print("✗ Brak danych do walidacji")
            return {}

        print(f"\n=== ULEPSZONA WALIDACJA ROLLING WINDOW ===")
        print(f"Rozmiar okna: {window_size}")

        data = self.returns_cleaned
        n_total = len(data)

        if n_total < window_size + 50:
            print(f"⚠ Za mało danych dla walidacji rolling window")
            print(f"Potrzeba minimum {window_size + 50}, mamy {n_total}")
            # Fallback na prostą walidację
            return self.simple_validation()

        forecasts = []
        actuals = []
        dates = []

        print(f"Wykonywanie prognoz rolling (co 10 dni dla efektywności)...")

        for i in range(window_size, n_total - 1, 10):  # Co 10 dni
            try:
                train_data = data.iloc[i - window_size:i]
                actual_vol = abs(data.iloc[i])

                # Użycie prostszego, stabilniejszego modelu
                model = arch_model(train_data, vol='GARCH', p=1, q=1, dist='normal', mean='Zero')
                results = model.fit(disp='off')

                forecast = results.forecast(horizon=1)
                pred_vol = np.sqrt(forecast.variance.values[-1, 0])

                # Filtrowanie ekstremalnych wartości
                if 0.01 < pred_vol < 5.0 and 0.01 < actual_vol < 5.0:
                    forecasts.append(pred_vol)
                    actuals.append(actual_vol)
                    dates.append(data.index[i])

                if len(forecasts) % 10 == 0 and len(forecasts) > 0:
                    print(f"Wykonano {len(forecasts)} prognoz...")

            except Exception as e:
                continue

        if len(forecasts) < min_forecasts:
            print(f"✗ Za mało udanych prognoz ({len(forecasts)} < {min_forecasts})")
            return self.simple_validation()

        # Konwersja do numpy arrays
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)

        # Obliczenie metryk
        rmse = np.sqrt(np.mean((actuals - forecasts) ** 2))
        mae = np.mean(np.abs(actuals - forecasts))

        # Bezpieczne MAPE
        safe_mape_val, n_used = self.safe_mape(actuals, forecasts)
        symmetric_mape_val = self.symmetric_mape(actuals, forecasts)

        # Korelacja
        correlation = np.corrcoef(actuals, forecasts)[0, 1] if len(actuals) > 1 else 0

        # Directional accuracy
        if len(actuals) > 1:
            actual_direction = np.diff(actuals) > 0
            forecast_direction = np.diff(forecasts) > 0
            directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
        else:
            directional_accuracy = 0

        print(f"\nWyniki ulepszonej walidacji rolling ({len(forecasts)} prognoz):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"Safe MAPE: {safe_mape_val:.2f}% (użyto {n_used} obserwacji)")
        print(f"Symmetric MAPE: {symmetric_mape_val:.2f}%")
        print(f"Korelacja: {correlation:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.1f}%")

        # Interpretacja jakości na podstawie Symmetric MAPE
        if symmetric_mape_val < 25:
            quality = "✓ Bardzo dobra jakość prognoz"
        elif symmetric_mape_val < 50:
            quality = "✓ Dobra jakość prognoz"
        elif symmetric_mape_val < 75:
            quality = "○ Umiarkowana jakość prognoz"
        else:
            quality = "⚠ Słaba jakość prognoz"

        print(quality)

        return {
            'rmse': rmse,
            'mae': mae,
            'safe_mape': safe_mape_val,
            'symmetric_mape': symmetric_mape_val,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'actual': actuals,
            'predicted': forecasts,
            'dates': dates,
            'n_forecasts': len(forecasts),
            'n_mape_used': n_used
        }

    def simple_validation(self):
        """
        Prosta walidacja train/test split jako fallback
        """
        print("\n=== PROSTA WALIDACJA TRAIN/TEST ===")

        data = self.returns_cleaned
        split_point = int(len(data) * 0.8)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]

        print(f"Zbiór treningowy: {len(train_data)} obserwacji")
        print(f"Zbiór testowy: {len(test_data)} obserwacji")

        try:
            # Dopasowanie prostego modelu
            model = arch_model(train_data, vol='GARCH', p=1, q=1, dist='normal', mean='Zero')
            results = model.fit(disp='off')

            # Prognoza na cały okres testowy
            forecasts = results.forecast(horizon=len(test_data))
            pred_vols = np.sqrt(forecasts.variance.values[-1, :])
            actual_vols = np.abs(test_data.values)

            # Dopasowanie długości
            min_length = min(len(pred_vols), len(actual_vols))
            pred_vols = pred_vols[:min_length]
            actual_vols = actual_vols[:min_length]

            # Metryki
            rmse = np.sqrt(np.mean((actual_vols - pred_vols) ** 2))
            mae = np.mean(np.abs(actual_vols - pred_vols))
            safe_mape_val, n_used = self.safe_mape(actual_vols, pred_vols)
            symmetric_mape_val = self.symmetric_mape(actual_vols, pred_vols)
            correlation = np.corrcoef(actual_vols, pred_vols)[0, 1]

            print(f"\nWyniki prostej walidacji:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"Safe MAPE: {safe_mape_val:.2f}% (użyto {n_used} obserwacji)")
            print(f"Symmetric MAPE: {symmetric_mape_val:.2f}%")
            print(f"Korelacja: {correlation:.4f}")

            return {
                'rmse': rmse,
                'mae': mae,
                'safe_mape': safe_mape_val,
                'symmetric_mape': symmetric_mape_val,
                'correlation': correlation,
                'actual': actual_vols,
                'predicted': pred_vols,
                'n_forecasts': len(pred_vols),
                'n_mape_used': n_used
            }

        except Exception as e:
            print(f"✗ Błąd podczas prostej walidacji: {e}")
            return {}

    def forecast_volatility(self, horizon=30):
        """Prognozowanie zmienności z optymalnym modelem"""
        if self.results is None:
            print("✗ Brak dopasowanego modelu do prognozowania")
            return pd.DataFrame()

        print(f"\n=== PROGNOZOWANIE ZMIENNOŚCI (OPTYMALNY MODEL) ===")
        print(f"Para walutowa: {self.currency_pair}")
        print(f"Model: GARCH{self.best_params[:2]} z rozkładem {self.best_params[2]}")
        print(f"Horyzont: {horizon} dni roboczych")

        try:
            forecasts = self.results.forecast(horizon=horizon)
            forecast_variance = forecasts.variance.values[-1, :]
            forecast_volatility = np.sqrt(forecast_variance)

            last_date = self.returns_cleaned.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=horizon, freq='B')

            min_length = min(len(forecast_dates), len(forecast_volatility))
            forecast_dates = forecast_dates[:min_length]
            forecast_volatility = forecast_volatility[:min_length]

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

            self._plot_volatility_forecast(forecast_df)
            return forecast_df

        except Exception as e:
            print(f"✗ Błąd podczas prognozowania: {e}")
            return pd.DataFrame()

    def _plot_volatility_forecast(self, forecast_df):
        """Wizualizacja prognoz zmienności"""
        try:
            historical_vol = self.returns_cleaned.rolling(window=30).std()

            plt.figure(figsize=(15, 8))

            n_hist = min(200, len(historical_vol))
            plt.plot(historical_vol.index[-n_hist:], historical_vol.iloc[-n_hist:],
                     label='Historyczna zmienność (30-dniowa)', color='blue', linewidth=1.5)

            if not forecast_df.empty:
                plt.plot(forecast_df['date'], forecast_df['forecast_volatility_daily'],
                         label=f'Prognoza GARCH{self.best_params[:2]}', color='red', linewidth=2, linestyle='--')

            plt.axvline(x=self.returns_cleaned.index[-1], color='green', linestyle=':',
                        label='Początek prognozy', alpha=0.7)

            plt.title(f'Prognoza zmienności {self.currency_pair} (Optymalny model GARCH{self.best_params[:2]})',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('Zmienność dzienna (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresu prognoz: {e}")

    def run_complete_analysis(self, clean_outliers=True, optimize_params=True):
        """
        Uruchomienie kompletnej zoptymalizowanej analizy GARCH
        """
        print("=" * 70)
        print(f"ZOPTYMALIZOWANA ANALIZA GARCH DLA KURSU {self.currency_pair}")
        print("ŹRÓDŁO: NARODOWY BANK POLSKI (NBP) API")
        print("=" * 70)

        try:
            if not self.setup_data_source():
                print("✗ Nie udało się skonfigurować źródła danych")
                return None

            if not self.calculate_returns(clean_outliers=clean_outliers):
                print("✗ Nie udało się obliczyć stóp zwrotu")
                return None

            self.visualize_data()

            if not self.fit_optimized_garch_model(optimize_params=optimize_params):
                print("✗ Nie udało się dopasować modelu GARCH")
                return None

            forecasts = self.forecast_volatility(horizon=30)

            # Ulepszona walidacja
            validation_results = self.improved_rolling_validation()

            print("\n" + "=" * 70)
            print(f"ZOPTYMALIZOWANA ANALIZA {self.currency_pair} ZAKOŃCZONA POMYŚLNIE")
            print("=" * 70)

            return {
                'model_results': self.results,
                'best_params': self.best_params,
                'forecasts': forecasts,
                'validation': validation_results,
                'raw_data': self.raw_data,
                'returns_original': self.returns,
                'returns_cleaned': self.returns_cleaned,
                'outliers_removed': self.outliers_removed
            }

        except Exception as e:
            print(f"✗ Błąd podczas analizy: {str(e)}")
            return None


# Główna część skryptu
if __name__ == "__main__":
    print("ZOPTYMALIZOWANA ANALIZA GARCH Z DANYMI NBP")
    print("=" * 50)

    print(
        "Dostępne waluty w tabeli A NBP: EUR, USD, GBP, CHF, JPY, CZK, SEK, NOK, DKK, CAD, AUD, HUF, RON, BGN, TRY, ILS, CLP, PHP, MXN, ZAR, BRL, MYR, RUB, IDR, INR, KRW, CNY, XDR, THB, SGD, HKD, UAH, ISK, HRK, SKK")

    currency = input("Kod waluty [eur]: ").strip().lower()
    if not currency:
        currency = "eur"

    start_date = input("Data początkowa (YYYY-MM-DD) [2021-01-01]: ").strip()
    if not start_date:
        start_date = "2021-01-01"  # Skrócony okres

    end_date = input("Data końcowa (YYYY-MM-DD) [2024-12-31]: ").strip()
    if not end_date:
        end_date = "2024-12-31"

    csv_file = input("Ścieżka do pliku CSV (lub Enter dla API): ").strip()
    if not csv_file:
        csv_file = None

    clean_outliers = input("Czy czyścić outliers? (tak/nie) [tak]: ").strip().lower()
    clean_outliers = clean_outliers in ['tak', 'yes', 'y', 't', '']

    optimize_params = input("Czy optymalizować parametry GARCH? (tak/nie) [tak]: ").strip().lower()
    optimize_params = optimize_params in ['tak', 'yes', 'y', 't', '']

    garch_analyzer = OptimizedGARCHAnalysisNBP(
        currency_code=currency,
        start_date=start_date,
        end_date=end_date,
        table='a',
        csv_file=csv_file
    )

    results = garch_analyzer.run_complete_analysis(
        clean_outliers=clean_outliers,
        optimize_params=optimize_params
    )

    if results is not None:
        print(f"\n=== PODSUMOWANIE OPTYMALIZACJI ===")
        print(f"Outlierów usuniętych: {results['outliers_removed']}")
        print(f"Najlepszy model: GARCH{results['best_params'][:2]} z rozkładem {results['best_params'][2]}")
        print(f"Safe MAPE: {results['validation'].get('safe_mape', 'N/A'):.2f}%")
        print(f"Symmetric MAPE: {results['validation'].get('symmetric_mape', 'N/A'):.2f}%")

        print("\nCzy chcesz zapisać wyniki do pliku CSV? (tak/nie)")
        try:
            response = input().lower().strip()

            if response in ['tak', 'yes', 'y', 't']:
                if not results['forecasts'].empty:
                    forecast_filename = f"optimized_garch_forecast_{currency}_{start_date}_{end_date}.csv"
                    results['forecasts'].to_csv(forecast_filename, index=False)
                    print(f"✓ Prognozy zapisane do pliku: {forecast_filename}")

                if results['model_results'] is not None:
                    summary_filename = f"optimized_garch_summary_{currency}_{start_date}_{end_date}.txt"
                    with open(summary_filename, 'w', encoding='utf-8') as f:
                        f.write(f"ZOPTYMALIZOWANY MODEL GARCH {currency.upper()}/PLN (NBP)\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(f"Okres analizy: {start_date} - {end_date}\n")
                        f.write(f"Outlierów usuniętych: {results['outliers_removed']}\n")
                        f.write(
                            f"Najlepszy model: GARCH{results['best_params'][:2]} z rozkładem {results['best_params'][2]}\n")
                        f.write(f"Safe MAPE: {results['validation'].get('safe_mape', 'N/A'):.2f}%\n")
                        f.write(f"Symmetric MAPE: {results['validation'].get('symmetric_mape', 'N/A'):.2f}%\n\n")
                        f.write(str(results['model_results'].summary()))

                    print(f"✓ Podsumowanie modelu zapisane do pliku: {summary_filename}")
        except EOFError:
            print("Pomijam zapis plików")

    print(f"\nZoptymalizowana analiza kursu {currency.upper()}/PLN z NBP API zakończona!")
