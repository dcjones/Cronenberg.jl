
# This is just a wrapper for the python code. It might be nice to rewrite this
# in julia someday, but it would be a pain and probably end up running slower.


function fit_expression_model(
        input_h5ad_filename::String,
        output_params_filename::String, seed::Int=0)

    adata = anndata_py.read_h5ad(input_h5ad_filename)
    cvae_py.fit_cvae(
        adata, adata.obs.label, output_params_filename, seed=seed)
end


function sample_expression_model(
        input_params_filename::String,
        input_h5ad_filename::String,
        output_h5ad_filename::String)

    adata = anndata_py.read_h5ad(input_h5ad_filename)
    cvae_py.sample_cvae(
        adata, adata.obs.label-1, input_params_filename, output_h5ad_filename)
end
