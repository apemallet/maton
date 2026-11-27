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
          shellHook = ''
            # CUDA JAX needs the host's libcuda: WSL2 exposes the Windows driver here.
            for lib in /usr/lib/wsl/lib /run/opengl-driver/lib; do
              [ -d "$lib" ] && export LD_LIBRARY_PATH="$lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            done
            # A checkout shared with Windows (/mnt/...) keeps the Linux venv out of its .venv.
            case "$PWD" in
              /mnt/*) export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/$(basename "$PWD")" ;;
            esac
          '';
        };
      }
    );
}
