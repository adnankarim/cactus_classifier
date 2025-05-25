// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", subtitle: "", authors: (), date: "", body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)
  set page(
    numbering: "1/1",
    number-align: right,
    header: context {
      if counter(page).get().first() > 1 [
        _#title _
        #h(1fr)
        _#subtitle _
      ]
    }
  )
  set text(font: "New Computer Modern")
  set list(marker: [sym.bullet])

  // Title row.
  align(center)[
    #block(
      text(weight: 700, 1.75em, title),
      width: 35em
    )
  ]

  // Subtitle row.
  align(center)[
    #block(
      text(weight: 500, 1.75em, subtitle),
      width: 40em
    )
  ]

  // Author information.
  pad(
    top: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, strong(author))),
    ),
  )

  // Date
  pad(
    top: 0em,
    bottom: 0.5em,
    x: 2em,
    align(center, date)
  )

  set heading(numbering: "1.1.")
  show heading.where(level: 2): it => block({
    counter(heading).display()
    h(8pt)
    it.body
  })

  show heading.where(level: 1): it => block({
    counter(heading).display()
    h(12pt)
    it.body
    v(2pt)
  })
  
  show heading.where(level: 3): it => block({
    counter(heading).display()
    h(12pt)
    it.body
    v(2pt)
  })

  set list(
    marker: ([--], [--]),
    spacing: 1em,
    indent: 8pt,
    body-indent: 8pt
  )

  // Main body.
  set par(justify: true)

  body
}
