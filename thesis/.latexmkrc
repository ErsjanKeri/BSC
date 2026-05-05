# From https://tex.stackexchange.com/questions/58963/latexmk-with-makeglossaries-and-auxdir-and-outdir#59098
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
sub makeglossaries {
  my ($base_name, $path) = fileparse($_[0]);
  pushd $path;
  my $return = system "makeglossaries $base_name";
  popd;
  return $return;
}

$success_cmd = 'make _fachschaft-print';

# This thesis uses biblatex with backend=biber. Force latexmk to invoke biber
# (not bibtex) so `make pdf` runs end-to-end without dumping artifacts at root
# when the bibtex step fails. latexmk's default biber command already handles
# the output directory through its own bookkeeping; we just need to enable it.
$bibtex_use = 2;
