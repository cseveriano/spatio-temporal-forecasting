import pandas as pd
import mySSA

def ssa_decompose(df):
    p_inds = [i for i in range(3)]
    df_clean = pd.DataFrame(columns=df.columns)
    df_residual = pd.DataFrame(columns=df.columns)

    for c in df.columns:
        dfc = df[c]
        cl = []
        rs = []
        for date in df.index.to_period('M').unique():
            ts = dfc[str(date)]
            N = int(len(ts))  # number of samples
            T = 96  # sample daily frequency (4 samples per hour)
            embedding_dimension = int(N / T)
            ssa = mySSA(ts)
            ssa.embed(embedding_dimension=embedding_dimension, verbose=True)
            res_streams = [j for j in range(3, embedding_dimension)]
            ssa.decompose(verbose=True)
            principal = ssa.view_reconstruction(*[ssa.Xs[i] for i in p_inds], names=p_inds, plot=False,
                                                return_df=True)
            residual = ssa.view_reconstruction(*[ssa.Xs[i] for i in res_streams], names=res_streams, plot=False,
                                               return_df=True)

            cl.append(principal.values)
            rs.append(residual.values)
            del ssa
        df_clean[c] = cl
        df_residual[c] = rs

    return df_clean, df_residual