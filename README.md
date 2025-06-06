# PFD Toolkit <a href='https://github.com/sam-osian/pfd-toolkit'><img src='docs/assets/badge.png' align="right" height="120" /></a>

Turn raw PFD reports into structured insights — fast.

PFD Toolkit is a suite of tools that replaces the manual effort involved in the collection, screening, and thematic discovery of PFD reports. It helps researchers, journalists, and public health analysts turn raw reports into actionable insights.



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
    category='all',
    start_date="2024-01-01",
    end_date="2025-01-01"
)
```


`reports` will be a pandas DataFrame. Each row is a separate report, and each column is a report section. For example:


| url                                   | id         | date       | coroner_name      | area                           | receiver                  | investigation           | circumstances       | concerns         |
|----------------------------------------|------------|------------|------------------|--------------------------------|---------------------------|-----------------------------------|----------------------------|--------------------------|
| https://www.judiciary.uk/prevention...<br> | 2025-0207 | 2025-04-30 | Alison Mutch     | Manchester South               | Flixton Road...        | On 1st October...                 | Louise Danielle...         | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0208 | 2025-04-30 | Joanne Andrews   | West Sussex...       | West Sussex County...     | On 02 November...                 | Mrs Turner drove...        | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0120 | 2025-04-25 | Mary Hassell     | Inner North London             | The President...       | On 23 August...                   | Jan was a big baby...      | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0206 | 2025-04-25 | Jonathan Heath   | North Yorkshire and York       | Townhead Surgery          | On 04 June...                     | On 15 March 2024...        | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0199 | 2025-04-24 | Samantha Goward  | Norfolk                        | The Department...         | On 22 August...                   | In summary, on...          | During the course...     |

For more information on each of these columns / report sections, please see the package documentation. 


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
- Some project dependencies may have their own licence terms, which could affect certain types of use.
- If you plan to use this software in a commercial or proprietary setting, please review all relevant licences to ensure compliance.