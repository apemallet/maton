{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            uv
            just
            # Snakemake comes from uv's dev group; nixpkgs' copy leaks a
            # Python 3.14 PYTHONPATH into the 3.13 venv and breaks pytest.
          ];
        };
      }
    );
}
