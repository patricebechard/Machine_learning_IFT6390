import bibtexparser
with open('report.tex') as bibtex_file:
    parser = bibtexparser.bparser.BibTexParser()
    parser.customization = bibtexparser.cutsomization.convert_to_unicode
    bibliography = bibtexparser.load(bibtex_file, parser=parser)
    for entry in bibliography.entries:
        if entry.has_key('file'):
            shutil.copy(entry['file'], …)