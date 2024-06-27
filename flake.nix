{
  description = "Sets up the dependencies for developing verus";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem(system:
      let
      	pkgs     = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
      	version  = "0.0.1";
      	src      = self;
      in rec {      
        devShell = pkgs.mkShell {
          shellHook = ''
            SHELL=${pkgs.bashInteractive}/bin/bash
            VERUS_Z3_PATH=$(whereis z3 | awk '{print $2}')
          '';
          buildInputs = [
            pkgs.bashInteractive
          ];
          nativeBuildInputs = with pkgs; [
            rustup
            (vscode-with-extensions.override {
              vscodeExtensions = with pkgs.vscode-extensions; [
                rust-lang.rust-analyzer
              ];
            })


            tokei
      
            llvmPackages_14.libcxxStdenv
            llvmPackages_14.libunwind
            llvmPackages_14.libcxx
            clang-tools_14
            gnumake

            z3_4_12
          ];
        };
      }

    );

}

