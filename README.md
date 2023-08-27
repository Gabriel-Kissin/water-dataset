# water-dataset

The idea of this notebook is to create a small, simple classification dataset as an alternative to the well-known iris and penguins datasets.  

The raw data is in the `data` folder, and originates from the [UK Environment Agency website](https://environment.data.gov.uk/water-quality/view/download). This is processed step-by-step in the `water dataset.ipynb` notebook, with clear explanations of what is being done at each step. Various challenges (e.g. the uncompressed size of the full dataset being over 20 GB) and how they were surmounted are also discussed in the notebook. Other decisions made, such as which classes to use, and which determinands to include, are also explained. The final small dataset is saved as a `.csv` file (`water.csv`) and as a pickle of the pandas DataFrame (`water.pkl`). Some visualisation of the data is available at the end of the notebook, together with some interpretations of the plots, as well as some classifiers. However this is intentionally brief, as the main purpose of the notebook is to show the ETL process, rather than to actually analyse the data, which is left to the reader.

The classification categories are various fresh water types (sewage water types and seawater types were excluded), as follows:

<table style="table-layout: fixed ; width: 100%;">
  <tr>
    <th>River water</th>
    <th>Canal water</th>
    <th>Lake water</th>
    <th>Groundwater</th>
    <th>Estuary water</th>
  </tr>
  <tr>
    <td style='text-align:center; vertical-align:middle'><img src="images/A River Bank (The Seine at Asnières) - Seurat.jpg" width=200 height=150></td>
    <td style='text-align:center; vertical-align:middle'><img src="images/A Regatta on the Grand Canal - Canaletto.jpg" width=200 height=150></td>
    <td style='text-align:center; vertical-align:middle'><img src="images/Lakeside Landscape - Renoir.jpg" width=200 height=150></td>
    <td style='text-align:center; vertical-align:middle'><img src="images/At the Well - Edward Bird.jpg" width=200 height=150></td>
    <td style='text-align:center; vertical-align:middle'><img src="images/Thames Painting - The Estuary (Mouth of the Thames) - Michael Andrews.avif" width=200 height=150></td>
  </tr>
  <tr>
    <td><i> A River Bank (The Seine at Asnières) </i></td>
    <td><i> A Regatta on the Grand Canal </i></td>
    <td><i> Lakeside Landscape </i></td>
    <td><i> At the Well </i></td>
    <td><i> Thames Painting - The Estuary </i></td>
  </tr>
  <tr>
    <td> Georges Seurat </td>
    <td> Canaletto </td>
    <td> Pierre-Auguste Renoir </td>
    <td> Edward Bird </td>
    <td> Michael Andrews </td>
  </tr>
  <tr>
    <td> 1883 </td>
    <td> 1740 </td>
    <td> 1889 </td>
    <td> c. 1800 </td>
    <td> 1995 </td>
  </tr>
</table>

(Groundwater is the water present beneath Earth's surface in rock and soil pore spaces and in the fractures of rock formations, and is often withdrawn via wells.)

The columns are an index column (which can be deleted), the area from which the sample originated, the label of the exact sampling point location, the year over which the observation was averaged, and then the 6 observation features which are the basis for classification:
- Chloride [mg/l]
- Nitrite as N [mg/l]
- Nitrate as N [mg/l]
- Oxygen, Dissolved, % Saturation [%]
- pH [phunits]
- Temperature of Water [cel].

