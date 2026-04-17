import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class AdasynGaussian:

    @staticmethod
    def generate_synthetic_samples(X, y, minority_class, k=5, beta=1.0,
                                   d_threshold=0.5, epsilon=1e-6, alpha_blend=0.5):
        """
        ADASYN-Gaussian: ADASYN with Gaussian sampling instead of
        linear interpolation.

        Key differences from standard ADASYN:
        1. Adaptive density weighting (same as ADASYN) — minority samples
           surrounded by more majority class samples receive more
           synthetic samples.
        2. Gaussian sampling — samples are generated from a local neighborhood-based Gaussian distribution
           N(mu, sigma) rather than linear interpolation.
        3. Covariance = a blend of local covariance + global covariance of the
           minority class, resulting in meaningful variance.

        Parameters
        ----------
        X             : array (n_samples, n_features)
        y             : array (n_samples,)
        minority_class: label of the minority class
        k             : number of nearest neighbors
        beta          : balancing factor (1.0 = fully balanced)
        d_threshold   : imbalance threshold
        epsilon       : numerical regularization
        alpha_blend   : global covariance weight (0=local only, 1=global only)
        """
        minority_idx = np.where(y == minority_class)[0]
        X_minority = X[minority_idx]

        m_s = len(X_minority)
        m_l = np.sum(y != minority_class)
        d = m_s / m_l

        if d >= d_threshold:
            return X, y, np.empty((0, X.shape[1]))

        G = int((m_l - m_s) * beta)
        if G == 0:
            return X, y, np.empty((0, X.shape[1]))

        # ── 1. ADAPTIVE DENSITY WEIGHTING (ADASYN Core) ──────────────────────
        # ri = the proportion of majority neighbors surrounding each minority sample.
        # The higher ri is, the harder it is to classify → more
        # synthetic samples are needed.
        k_all = min(k, len(X) - 1)
        nn_all = NearestNeighbors(n_neighbors=k_all + 1).fit(X)

        r_i = np.zeros(m_s)
        for i in range(m_s):
            nbrs = nn_all.kneighbors([X_minority[i]], return_distance=False)[0]
            nbrs = [n for n in nbrs if n != minority_idx[i]][:k_all]
            majority_count = np.sum(y[nbrs] != minority_class)
            r_i[i] = majority_count / k_all

        r_sum = np.sum(r_i)
        if r_sum == 0:
            # All minority samples are surrounded by others — uniform distribution
            r_i = np.ones(m_s) / m_s
        else:
            r_i = r_i / r_sum  # normalize

        # gi = number of synthetic samples per minority sample (adaptive)
        g_i = np.round(r_i * G).astype(int)

        # ── 2. GLOBAL COVARIANCE FOR MINORITY CLASSES  ──────────────────────────────
        # Used as regularization to prevent the Gaussian variance from approaching zero.
        if m_s > 1:
            sigma_global = np.cov(X_minority, rowvar=False)
            if sigma_global.ndim == 0:
                sigma_global = np.eye(X.shape[1]) * float(sigma_global)
        else:
            sigma_global = np.eye(X.shape[1]) * epsilon

        sigma_global = _make_psd(sigma_global, epsilon)

        # ── 3. GAUSSIAN SAMPLING ─────────────────────────────────────────────
        # Apply Nearest Neighbors only to the minority class for local sampling.
        k_min = min(k, m_s - 1) if m_s > 1 else 1
        nn_min = NearestNeighbors(n_neighbors=k_min + 1).fit(X_minority)

        synthetic_samples = []

        for i in range(m_s):
            if g_i[i] == 0:
                continue

            nbrs_min = nn_min.kneighbors([X_minority[i]], return_distance=False)[0]
            nbrs_min = [n for n in nbrs_min if n != i][:k_min]
            local_pts = np.vstack([X_minority[nbrs_min], X_minority[i:i+1]])

            # Local average as a distribution hub
            mu = np.mean(local_pts, axis=0)

            # Local covariance (from the nearest neighbors of the minority class)
            if len(local_pts) > 1:
                sigma_local = np.cov(local_pts, rowvar=False)
                if sigma_local.ndim == 0:
                    sigma_local = np.eye(X.shape[1]) * float(sigma_local)
            else:
                sigma_local = np.zeros((X.shape[1], X.shape[1]))

            # Blend: sigma = (1-alpha)*sigma_lokal + alpha*sigma_global
            # alpha_blend > 0 ensures meaningful variance even when neighbors are very similar
            sigma = (1.0 - alpha_blend) * sigma_local + alpha_blend * sigma_global
            sigma = _make_psd(sigma, epsilon)

            for _ in range(g_i[i]):
                s = np.random.multivariate_normal(mu, sigma, check_valid='ignore')
                synthetic_samples.append(s)

        if len(synthetic_samples) > 0:
            synthetic_samples = np.array(synthetic_samples)
            X_synthetic = np.vstack([X, synthetic_samples])
            y_synthetic = np.hstack([y, [minority_class] * len(synthetic_samples)])
            return X_synthetic, y_synthetic, synthetic_samples

        return X, y, np.empty((0, X.shape[1]))

    @staticmethod
    def evaluate_k(X, y, k_values, beta_values, minority_class, **kwargs):
        results = []
        for k in k_values:
            for beta in beta_values:
                X_synthetic, y_synthetic, _ = AdasynGaussian.generate_synthetic_samples(
                    X, y, minority_class, k=k, beta=beta, **kwargs
                )
                knn = KNeighborsClassifier()
                knn.fit(X_synthetic, y_synthetic)
                y_pred = knn.predict(X)
                f1 = f1_score(y, y_pred, average='weighted')
                results.append((k, beta, f1))
        return results


def _make_psd(sigma, epsilon):
    """Pastikan matriks simetris dan positive semi-definite."""
    sigma = (sigma + sigma.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, epsilon)
    sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    sigma = (sigma + sigma.T) / 2 + np.eye(sigma.shape[0]) * epsilon
    return sigma
