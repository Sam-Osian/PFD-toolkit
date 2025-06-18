# PFD Toolkit <a href='https://github.com/sam-osian/pfd-toolkit'><img src='docs/assets/badge.png' align="right" height="120" /></a>

Turn raw PFD reports into structured insights — fast.

PFD Toolkit is a suite of tools that replaces the manual effort involved in the collection, screening, and thematic discovery of PFD reports. It helps researchers, journalists, and public health analysts turn raw reports into actionable insights.

For more information, please consult the [documentation](https://sam-osian.github.io/PFD-toolkit/).

## Getting started


### Installation


```bash
pip install pfd_toolkit
```


### Load PFD Report Data (in seconds)

To load PFD data, just import the module, specify the category of reports and your date-range:

```python
from pfd_toolkit import load_reports

reports = load_reports(
    start_date="2024-01-01",
    end_date="2025-01-01"
)
```

`reports` will be a pandas DataFrame. Each row is a separate report, and each column is a report section. For example:


| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


For more information on each of these columns / report sections, please see the [documentation](https://sam-osian.github.io/PFD-toolkit/). 


### Update reports

PFD Toolkit updates each week with freshly published reports. To access these new reports, you will need to update the Toolkit:

```bash
pip install --upgrade pfd_toolkit
```



### Key features

Beyond loading reports, PFD Toolkit lets you:
 * Screen reports: find cases relevant to your specific research question.
 * Summarise text: distill full-length reports into a custom summary.
 * Discover themes: uncover recurring topics contained within a selection of reports.
 * Tabulate reports: categorise & tabulate reports by discovered themes — or provide your own themes.

To get started with these features, please check out our documentation.


## Licence

This project is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](./LICENCE) file for the full text.

**Please note:**
- You are welcome to use, modify, and share this code under the terms of the AGPL-3.0.
- If you use this code to provide a networked service, you are required to make the complete source code available to users of that service.
- Some project dependencies may have their own licence terms, which could affect certain types of use (e.g. commercial use). Please review all relevant licences to ensure compliance.