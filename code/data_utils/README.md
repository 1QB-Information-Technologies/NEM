# Data_utils

`MeasurementsDataset`: The formatting class for measurements results that are going to be used in the NQSTomographyTrainer. Here, the measurements are organized by (a) the measurement bases and (b) their outcomes.

`MeasurementsInCBDataset`: The formatting class for measurements after they have been transformed into the computational basis. Each measurement outcome is transformed into the computational basis into (A) sampled states and the (B) amplitudes. NQSTomographyTrainer takes measurements in the computational basis.  

`MeasurementsInCBDataLoader`: DataLoader for NQSTomographyTrainer where the measurements are transformed into the computational basis

`TomographyMeasurements`: The TomographyMeasurements class create the measurement circuits needed to
    perform the measurements in the specified bases chosen for NQST. It also
    reformats the bases/samples into a dictionary that is easier to convert
    into a MeasurementsDataset (as done in circuit_samples_to_MeasurementsDataset)
