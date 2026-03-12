

----------- v1.3 -----------
- Powered by Multi-Task DeepHit v2.14.
- Added and refined simultaneous uncertainty band workflow for mortality and trajectories.
- Mortality summary cards updated to 3-year / 5-year / 8-year / 10-year risk.
- Summary cards now show interval bounds in bracket format when available, e.g. `x% [lower%, upper%]`.
- Mobile header refinement: logo is right-aligned in mobile mode.

----------- v1.2.2 -----------
- Powered by Multi-Task DeepHit v2.14.
- Fixed the issue : ImportError: libgomp.so.1: cannot open shared object file: No such file or directory.
- Moved the app deployment to Posit Connect Cloud.
- Redesigned the user interface with NHLBI official branding colors (#003087 navy, #0067B1 blue, #C8102E red).
- Added NHLBI logo to the header banner.
- Improved mobile responsiveness: layout adapts to phone and tablet screen sizes using Bootstrap breakpoints and CSS media queries.

----------- v1.2.1 -----------
- Powered by Multi-Task DeepHit v2.14.
- Placed codes for model to utils.
- Organized required files and saved into utils folder.

----------- v1.2.0 -----------
- Powered by Multi-Task DeepHit v2.14. Introduced temporal weighting in the longitudinal loss to make the model more sensitive to early changes in risk factors.
- Added conformal prediction intervals for mortality risk.
- Introduced a calibration data upload module for computing conformal scores.
- Added an adjustable alpha slider for users to control the conformal prediction level.
- Implemented format validation for uploaded patients data and calibration data to detect and reject incompatible files.
- Refined the decimal formatting of mortality predictions for improved readability.
- Added option to download the prediction table as a CSV file.
- Improved the user interface.

----------- v1.1.0 -----------

- Powered by Multi-Task Deephit v2.11.
- Added predicted trajectories of 12 risk factors over 3 years.
- Users can edit patient features to observe how both the mortality plot and risk factor trajectories change.
- Original and updated curves are displayed together for comparison.

----------- v1.0.0 -----------

- Model powered by Multi-Task Deephit v2.11.
- Users can download example data and upload patient data with the same structure.
- The app provides 15-year mortality prediction and cumulative mortality plots.
- SHAP waterfall plots highlight the most influential risk factors for 5-year mortality. Users can click on a patient row to view the contribution of each risk factor.
- Users can edit patient features to check how predicted 5-year mortality and feature importance change accordingly.