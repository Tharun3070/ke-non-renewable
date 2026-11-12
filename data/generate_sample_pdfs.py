"""Generate multi-page sample renewable energy PDF documents covering all 8 knowledge domains."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"


def _build_pdf(
    filename: str,
    doc_title: str,
    pages: list[dict[str, list[str] | str]],
    overwrite: bool = False,
) -> Path:
    """Render a structured multi-page PDF with headings and detailed technical paragraphs."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DOCUMENTS_DIR / filename
    if output_path.exists() and not overwrite:
        return output_path

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=14,
        textColor="#0B2545",
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=8,
        textColor="#1B7F4C",
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=8,
    )

    story = [Paragraph(doc_title, title_style), Spacer(1, 0.1 * inch)]

    for page_idx, page_spec in enumerate(pages):
        if page_idx > 0:
            story.append(PageBreak())
        story.append(Paragraph(str(page_spec["heading"]), heading_style))
        for para in page_spec["paragraphs"]:
            story.append(Paragraph(str(para), body_style))

    pdf_doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=doc_title,
    )
    pdf_doc.build(story)
    return output_path


def generate_all_sample_pdfs() -> list[Path]:
    """Create the four multi-page sample renewable energy PDFs."""
    created: list[Path] = []

    created.append(
        _build_pdf(
            "01_solar_and_wind_energy_systems.pdf",
            "Solar and Wind Energy Systems: Principles, Technologies, and Performance",
            [
                {
                    "heading": "1. Solar Energy and Photovoltaic (PV) Systems",
                    "paragraphs": [
                        "Solar energy harnesses electromagnetic radiation from the sun using two primary pathways: "
                        "Solar Photovoltaics (PV), which convert sunlight directly into DC electricity via the photovoltaic effect, "
                        "and Concentrated Solar Power (CSP), which focuses sunlight using mirrors onto a thermal receiver to drive steam turbines.",
                        "Commercial crystalline silicon solar modules dominate global deployment. Monocrystalline PERC and TOPCon cells "
                        "achieve commercial module efficiencies between 20% and 23.5%, whereas polycrystalline silicon panels typically deliver 15% to 17%. "
                        "Emerging perovskite-silicon tandem cells have exceeded 33% efficiency in laboratory testing.",
                        "A complete grid-tied PV system consists of solar arrays, Maximum Power Point Tracking (MPPT) charge controllers or string inverters, "
                        "and bidirectional net meters. MPPT algorithms continuously adjust electrical operating voltage and current so panels deliver peak power "
                        "under varying irradiance and cell temperature conditions. High ambient temperatures reduce open-circuit voltage, typically lowering output by "
                        "0.30% to 0.45% per degree Celsius above 25 C.",
                        "Utility-scale solar plants increasingly use single-axis horizontal trackers, increasing annual energy yield by 15% to 25% compared with fixed-tilt arrays. "
                        "Bifacial modules capture both direct front-side sunlight and ground-reflected albedo irradiance on the rear surface, adding another 5% to 12% energy gain.",
                    ],
                },
                {
                    "heading": "2. Wind Energy Conversion and Turbine Engineering",
                    "paragraphs": [
                        "Wind turbines convert the kinetic energy of moving air into mechanical shaft torque and then electrical power. "
                        "According to Betz's Law, the theoretical maximum fraction of kinetic energy any wind turbine can extract from an open air stream is 59.3% "
                        "(16/27), known as the Betz limit. Modern three-bladed horizontal-axis wind turbines (HAWTs) achieve aerodynamic power coefficients of 0.45 to 0.50.",
                        "Available wind power scales linearly with air density and swept rotor area, and cubically with wind speed: P = 0.5 * rho * A * v^3 * Cp. "
                        "Because power varies with the cube of wind velocity, doubling wind speed increases theoretical available power eightfold. "
                        "Turbines start generating at a cut-in speed around 3 to 4 m/s, reach rated power near 11 to 14 m/s, and shut down at a cut-out speed of 25 m/s to prevent structural damage.",
                        "Onshore wind turbines typically feature hub heights of 80 to 130 meters and rated capacities of 3 MW to 6 MW, achieving capacity factors of 30% to 45%. "
                        "Offshore wind turbines benefit from stronger, steadier marine winds, scaling to 12 MW to 15 MW per unit with rotor diameters exceeding 220 meters and capacity factors of 45% to 60%. "
                        "Floating offshore platforms allow deployment in deep waters exceeding 60 meters where fixed monopiles are impractical.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "02_hydro_geothermal_biomass_energy.pdf",
            "Hydroelectric, Geothermal, and Biomass Renewable Energy Technologies",
            [
                {
                    "heading": "1. Hydroelectric Energy Systems",
                    "paragraphs": [
                        "Hydroelectric power generates electricity by converting the gravitational potential energy of elevated water into mechanical rotation in hydraulic turbines. "
                        "The three main architectures are impoundment (dam-and-reservoir) plants, run-of-river facilities that divert natural river flow without massive reservoirs, "
                        "and pumped-storage hydropower (PSH) plants that act as large-scale grid batteries.",
                        "Turbine selection depends on hydraulic head and flow rate. Pelton impulse turbines are suited for high-head, low-flow mountain sites (head > 300 m). "
                        "Francis reaction turbines serve medium-head installations (30 m to 300 m) and represent the most widely installed hydroelectric turbine globally. "
                        "Kaplan axial-flow propeller turbines operate efficiently in low-head, high-flow river sites (head < 30 m).",
                        "Hydropower provides exceptional grid inertia, rapid black-start capability, and dispatchable peaking power with plant efficiencies around 85% to 90%.",
                    ],
                },
                {
                    "heading": "2. Geothermal Energy Extraction",
                    "paragraphs": [
                        "Geothermal energy taps natural thermal reservoirs stored inside Earth's crust, providing firm, weather-independent baseload electricity and direct district heating "
                        "with capacity factors frequently exceeding 90%.",
                        "Three power-plant cycles are used depending on reservoir fluid temperature: Dry Steam plants pipe natural steam directly into a turbine (above 235 C); "
                        "Flash Steam plants depressurize high-temperature hot water (> 180 C) into steam separators; and Binary Cycle plants transfer heat from moderate-temperature geothermal brine "
                        "(100 C to 175 C) to a secondary organic working fluid with a low boiling point in a closed Organic Rankine Cycle loop, releasing virtually zero non-condensable emissions.",
                        "Enhanced Geothermal Systems (EGS) create engineered permeability in hot dry rock through controlled hydraulic stimulation, expanding geothermal viability beyond volcanic tectonic belts.",
                    ],
                },
                {
                    "heading": "3. Biomass and Bioenergy Pathways",
                    "paragraphs": [
                        "Biomass energy derives renewable heat, electricity, and transport fuels from organic feedstocks such as agricultural crop residues, forestry waste, dedicated energy crops, "
                        "municipal organic waste, and animal manure. When sustainably harvested so regrowth absorbs released carbon, bioenergy contributes to circular carbon management.",
                        "Biochemical conversion uses microorganisms in anaerobic digestion to break down wet organic matter in oxygen-free digesters, producing biogas composed of roughly 55% to 70% methane (CH4) and 30% to 45% CO2. "
                        "Upgrading biogas removes CO2 and trace hydrogen sulfide to yield pipeline-quality biomethane or compressed bio-CNG.",
                        "Thermochemical conversion includes direct combustion in combined heat and power (CHP) boilers, gasification (partial oxidation at 700 C to 1000 C producing syngas CO + H2), "
                        "and pyrolysis (thermal decomposition without oxygen yielding bio-oil and carbon-sequestering biochar).",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "03_green_hydrogen_and_energy_storage.pdf",
            "Green Hydrogen Production and Renewable Energy Storage Technologies",
            [
                {
                    "heading": "1. Green Hydrogen Production via Water Electrolysis",
                    "paragraphs": [
                        "Green hydrogen is produced by splitting water (H2O) into hydrogen (H2) and oxygen (O2) using electrolyzers powered exclusively by renewable electricity such as solar or wind. "
                        "Unlike grey hydrogen made via steam methane reforming (SMR) from natural gas, green hydrogen produces zero direct carbon emissions during production.",
                        "Three major electrolyzer technologies exist: Alkaline Water Electrolyzers (AWE) use a liquid potassium hydroxide (KOH) electrolyte, offer low capital cost and mature gigawatt-scale operation at 60% to 70% efficiency, "
                        "but respond more slowly to rapid wind/solar fluctuations. Proton Exchange Membrane (PEM) electrolyzers use a solid polymer membrane, operate at high current densities with compact footprints and sub-second ramp rates "
                        "ideal for variable renewables, though they rely on platinum and iridium catalysts. Solid Oxide Electrolyzer Cells (SOEC) operate at 700 C to 850 C and achieve electrical efficiencies above 80% when paired with industrial waste heat.",
                        "Producing 1 kilogram of green hydrogen requires roughly 50 to 55 kWh of electricity and 9 liters of purified water. Green hydrogen is essential for decarbonizing hard-to-abate sectors including green steelmaking, "
                        "green ammonia fertilizer synthesis, heavy maritime shipping, and sustainable aviation fuels.",
                    ],
                },
                {
                    "heading": "2. Renewable Energy Storage Systems",
                    "paragraphs": [
                        "Because solar and wind generation fluctuate with weather and daylight cycles, energy storage systems are critical for grid stability, peak shifting, and curtailment reduction.",
                        "Lithium-ion Battery Energy Storage Systems (BESS) lead short-to-medium duration grid storage (2 to 6 hours). Lithium Iron Phosphate (LFP) chemistry has become dominant for stationary grid storage over Nickel Manganese Cobalt (NMC) "
                        "due to superior thermal safety, cobalt-free supply chains, 4,000 to 6,000+ cycle life, and round-trip efficiencies of 85% to 92%.",
                        "For long-duration energy storage (8 to 24+ hours), Vanadium Redox Flow Batteries (VRFB) decouple power rating from energy capacity using external electrolyte tanks and achieve over 15,000 cycles without capacity degradation. "
                        "Sodium-ion batteries offer a low-cost lithium-free alternative using abundant sodium salts. Pumped Storage Hydropower (PSH) remains the largest installed storage technology worldwide by gigawatt-hour capacity, delivering 70% to 80% round-trip efficiency over 50-year asset lifetimes.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "04_government_renewable_energy_schemes.pdf",
            "Government Renewable Energy Schemes, Policies, and Incentive Frameworks",
            [
                {
                    "heading": "1. Major Indian Government Renewable Energy Schemes",
                    "paragraphs": [
                        "PM Surya Ghar: Muft Bijli Yojana (launched in 2024) targets rooftop solar installation across 10 million residential households in India, providing up to 300 units of free electricity per month. "
                        "Under the scheme, households receive Central Financial Assistance (CFA) subsidies of Rs 30,000 per kW for the first 2 kW and Rs 18,000 for the third kW, capped at Rs 78,000 for systems of 3 kW or larger.",
                        "PM-KUSUM (Pradhan Mantri Kisan Urja Suraksha evam Utthaan Mahabhiyan) supports agricultural decarbonization through three components: Component A builds decentralized 500 kW to 2 MW solar plants on barren farm land; "
                        "Component B installs standalone off-grid solar agricultural water pumps; and Component C solarizes existing grid-connected agricultural pumps so farmers can sell surplus power to state distribution utilities (DISCOMs).",
                        "The National Green Hydrogen Mission (NGHM) aims to establish India as a global hub for green hydrogen production and export, targeting at least 5 million metric tonnes (MMT) per annum of green hydrogen production capacity by 2030, "
                        "backed by roughly 125 GW of associated renewable energy capacity and SIGHT (Strategic Interventions for Green Hydrogen Transition) incentives for domestic electrolyzer manufacturing.",
                        "The Production Linked Incentive (PLI) Scheme for High Efficiency Solar PV Modules incentivizes gigawatt-scale domestic manufacturing from polysilicon to wafers, cells, and modules, reducing import dependence.",
                    ],
                },
                {
                    "heading": "2. International Policy Instruments and Grid Incentives",
                    "paragraphs": [
                        "Globally, governments combine tax credits, auctions, and grid-access mandates to accelerate clean energy transitions. In the United States, the Inflation Reduction Act (IRA) provides long-term Production Tax Credits (PTC) "
                        "and Investment Tax Credits (ITC) for solar, wind, standalone battery storage, and clean hydrogen production (Section 45V offering up to $3 per kg for ultra-low-carbon hydrogen).",
                        "The European Union's European Green Deal and REPowerEU framework target at least 42.5% renewable share in gross final energy consumption by 2030, streamlining permitting for solar, offshore wind, and renewable hydrogen valleys.",
                        "At the distribution level, Net Metering and Gross Metering policies allow prosumers with rooftop solar to export surplus electricity to the grid and receive billing credits or feed-in tariffs, improving project payback periods to 3 to 5 years.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "05_solar_photovoltaics_and_csp_engineering.pdf",
            "Solar Photovoltaics Physics, Inverters, Agrivoltaics, and Concentrated Solar Power",
            [
                {
                    "heading": "1. Photovoltaic Physics, Cell Architectures, and Plant Reliability",
                    "paragraphs": [
                        "Single-junction silicon solar cells are bounded by the Shockley-Queisser theoretical efficiency limit of approximately 33.7% under standard AM1.5G solar spectrum due to thermalization losses and sub-bandgap photon transmission. Silicon has an indirect bandgap of 1.12 eV, making it well suited for terrestrial solar conversion.",
                        "Utility and commercial solar installations rely on three inverter topologies: central inverters (1 MW to 5 MW) for flat utility fields, string inverters (20 kW to 350 kW) offering modular MPPT tracking across roof orientations, and module-level microinverters or DC optimizers that mitigate partial shading mismatch losses.",
                        "Long-term PV reliability depends on preventing Potential-Induced Degradation (PID), caused by high voltage stress between cells and grounded frames, and Light-Induced Degradation (LID) in boron-doped p-type wafers. Modern n-type TOPCon and Heterojunction (HJT) modules significantly reduce both LID and temperature coefficients.",
                        "Agrivoltaics combines elevated or spaced bifacial solar arrays with crop cultivation or sheep grazing underneath. Partial shading lowers soil evaporation and heat stress for shade-tolerant crops while panel cooling from crop evapotranspiration improves PV energy yield by 1% to 3%.",
                    ],
                },
                {
                    "heading": "2. Concentrated Solar Power (CSP) and Thermal Storage",
                    "paragraphs": [
                        "Concentrated Solar Power systems use optical reflectors to focus Direct Normal Irradiance (DNI) onto a heat-transfer receiver, generating high-temperature steam to drive conventional Rankine-cycle turbines. CSP performs best in arid high-DNI regions exceeding 2,000 kWh per square meter per year.",
                        "The two dominant utility CSP designs are Parabolic Trough collectors, which heat synthetic oil up to 390 C along linear focal tubes, and Solar Power Towers (Central Receivers), where thousands of two-axis heliostat mirrors concentrate sunlight onto a tower-mounted molten salt receiver reaching 565 C.",
                        "Integrating two-tank molten nitrate salt (60% NaNO3 and 40% KNO3) thermal energy storage enables CSP plants to dispatch solar electricity for 6 to 15 hours after sunset, delivering synchronous spinning inertia and firm evening peak power.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "06_offshore_wind_and_aerodynamics.pdf",
            "Offshore Wind Foundations, Aerodynamic Wake Modeling, and Drivetrain Engineering",
            [
                {
                    "heading": "1. Offshore Wind Foundations, Subsea Grids, and Wake Effects",
                    "paragraphs": [
                        "Offshore wind installations use seabed-fixed foundations in shallow-to-transitional waters: monopiles dominate depths up to 45 meters, while steel lattice jacket foundations serve 40 to 65 meter depths. Beyond 60 meters, floating platforms—semi-submersibles, spar buoys, and tension-leg platforms—unlock deep coastal wind resources.",
                        "Downstream turbines inside a wind farm experience reduced wind speed and higher turbulence known as the wake effect, which can reduce annual farm energy production by 8% to 15%. Modern farm layouts optimize inter-turbine spacing (7 to 10 rotor diameters) and apply active wake steering by intentionally yawing upstream rotors.",
                        "Nearshore offshore wind farms transmit AC power via 66 kV inter-array cables and 220 kV export cables, whereas distant arrays more than 80 km offshore use High-Voltage Direct Current (HVDC) converter platforms to eliminate reactive charging losses.",
                    ],
                },
                {
                    "heading": "2. Wind Turbine Drivetrains, Pitch Control, and Predictive Maintenance",
                    "paragraphs": [
                        "Wind turbines primarily employ either Doubly-Fed Induction Generators (DFIG) coupled with multi-stage gearboxes or gearless Direct-Drive Permanent Magnet Synchronous Generators (PMSG). Direct-drive PMSGs eliminate high-wear gearbox stages, lowering offshore maintenance costs.",
                        "Above rated wind speed, active blade pitch control rotates each blade around its longitudinal axis to shed excess aerodynamic torque and maintain constant generator speed and structural load limits.",
                        "Supervisory Control and Data Acquisition (SCADA) systems stream vibration, gearbox oil temperature, and bearing acoustic data at high frequency, enabling machine-learning anomaly detection to forecast bearing or blade-leading-edge erosion failures weeks in advance.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "07_hydropower_and_pumped_storage_engineering.pdf",
            "Hydropower Civil Engineering, Cavitation Control, and Pumped Storage Systems",
            [
                {
                    "heading": "1. Hydraulic Design, Penstocks, and Cavitation Mitigation",
                    "paragraphs": [
                        "In hydroelectric power plants, steel or concrete-lined penstocks convey pressurized water from the intake reservoir to the turbine spiral casing. Sudden valve closures during load rejection create water-hammer pressure surges, which are absorbed by vertical surge tanks or pressure-relief bypass valves.",
                        "Hydraulic turbine selection uses specific speed (Ns), a dimensionless index combining flow rate, rotational speed, and net hydraulic head. Low-specific-speed Pelton runners operate under high heads, medium-specific-speed Francis runners balance moderate head and flow, and high-specific-speed Kaplan runners handle high-volume low-head rivers.",
                        "Cavitation occurs when local static pressure near turbine runner blades drops below water vapor pressure, forming vapor bubbles that violently implode on metal surfaces and cause pitting erosion. Setting turbines at proper submergence depth (tailwater elevation) and using stainless-steel cavitation-resistant alloys prevent runner damage.",
                    ],
                },
                {
                    "heading": "2. Closed-Loop Pumped Storage Hydropower and Grid Stability",
                    "paragraphs": [
                        "Pumped Storage Hydropower (PSH) accounts for over 90% of global utility-scale electricity storage capacity. During surplus solar or wind hours, reversible Francis pump-turbines pump water from a lower reservoir to an upper reservoir; during peak demand, water flows back down to generate electricity.",
                        "Closed-loop PSH systems pair two off-stream artificial reservoirs or flooded former mines without damming natural flowing rivers, dramatically reducing ecological disruption and permitting timelines.",
                        "Variable-speed pumped-storage units powered by doubly-fed asynchronous machines allow active power regulation even during pumping mode, providing rapid frequency containment, synthetic inertia, and black-start grid restoration.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "08_geothermal_egs_and_district_heating.pdf",
            "Enhanced Geothermal Systems, Heat Pumps, and Direct-Use Thermal Networks",
            [
                {
                    "heading": "1. Enhanced Geothermal Systems (EGS) and Closed-Loop Drilling",
                    "paragraphs": [
                        "Conventional hydrothermal geothermal plants require natural heat, rock permeability, and subsurface water simultaneously. Enhanced Geothermal Systems (EGS) drill 3 to 5 km deep into hot dry crystalline rock (150 C to 300 C) and stimulate fracture networks via controlled fluid injection to circulate heat-transfer water between injection and production wells.",
                        "Closed-loop Advanced Geothermal Systems (AGS) circulate a working fluid continuously inside sealed downhole U-tube or multilateral wellbore heat exchangers, extracting conductive rock heat without hydraulic fracturing or fluid loss to surrounding formations.",
                        "Sustainable geothermal reservoir management relies on 100% spent-brine reinjection to maintain subsurface pore pressure, prevent surface subsidence, and avoid thermal or mineral contamination of shallow freshwater aquifers. Some geothermal brines in volcanic basins also support Direct Lithium Extraction (DLE) as a co-product.",
                    ],
                },
                {
                    "heading": "2. Geothermal District Heating and Ground Source Heat Pumps",
                    "paragraphs": [
                        "Direct-use geothermal district heating circulates 60 C to 100 C geothermal water through insulated municipal pipe networks and heat exchangers to warm residential buildings, greenhouses, and industrial food-drying facilities, achieving thermal conversion efficiencies above 80%.",
                        "Ground Source Heat Pumps (GSHP), also called geothermal heat pumps, exploit the stable year-round temperature (10 C to 18 C) at 2 to 150 meters underground using vertical boreholes or horizontal ground loops.",
                        "Because GSHPs move heat rather than creating it by combustion or resistance, they achieve Coefficients of Performance (COP) between 3.5 and 5.0, delivering 3.5 to 5 units of thermal heating or cooling per unit of electricity consumed.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "09_biomass_biofuels_and_waste_to_energy.pdf",
            "Advanced Biofuels, Biogas Upgrading, BECCS, and Biomass Waste-to-Energy",
            [
                {
                    "heading": "1. Second-Generation Biofuels and Sustainable Aviation Fuel",
                    "paragraphs": [
                        "First-generation biofuels rely on food crops such as sugarcane ethanol or soybean biodiesel, creating land-use competition. Second-generation (2G) cellulosic bioethanol converts non-food lignocellulosic residues—such as rice straw, wheat stubble, bagasse, and forestry trimmings—via steam pretreatment, enzymatic hydrolysis, and yeast fermentation.",
                        "Hydrotreated Vegetable Oil (HVO), or renewable diesel, processes used cooking oil, tallow, and non-edible oilseeds with hydrogen to yield drop-in hydrocarbon diesel chemically identical to fossil diesel with up to 85% lifecycle greenhouse gas reduction.",
                        "Sustainable Aviation Fuel (SAF) is produced primarily through Hydroprocessed Esters and Fatty Acids (HEFA) and Biomass Gasification + Fischer-Tropsch synthesis, enabling commercial aircraft to cut lifecycle aviation emissions without modifying jet engines.",
                    ],
                },
                {
                    "heading": "2. Biogas Upgrading, Biochar, and BECCS Negative Emissions",
                    "paragraphs": [
                        "Raw biogas from anaerobic digestion of agricultural manure, sewage sludge, and food waste contains 55% to 70% methane alongside CO2 and corrosive hydrogen sulfide (H2S). Upgrading technologies including Water Scrubbing, Pressure Swing Adsorption (PSA), and Polymer Membrane Separation purify methane above 96% to produce Compressed Biogas (CBG) or biomethane.",
                        "Bioenergy with Carbon Capture and Storage (BECCS) captures biogenic CO2 released during biomass combustion or ethanol fermentation and injects it into deep saline aquifers for permanent geological storage, generating net-negative carbon emissions.",
                        "Slow pyrolysis of crop residues at 450 C to 600 C produces porous carbon-rich biochar, which locks carbon in agricultural soils for centuries while improving soil water retention and nutrient exchange.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "10_green_hydrogen_ammonia_and_transport.pdf",
            "Green Hydrogen Storage, Transport, Green Ammonia, and Industrial Decarbonization",
            [
                {
                    "heading": "1. Hydrogen Compression, Liquefaction, and Geological Storage",
                    "paragraphs": [
                        "Green hydrogen has the highest gravimetric energy density of any chemical fuel (120 MJ/kg lower heating value, nearly triple gasoline), but very low volumetric density at ambient pressure (0.089 kg per cubic meter), requiring compression, liquefaction, or chemical carriers for storage and transport.",
                        "For mobility and modular storage, gaseous hydrogen is compressed to 350 bar for heavy-duty buses/trucks or 700 bar for passenger fuel-cell vehicles in lightweight carbon-fiber-wrapped Type IV composite pressure vessels. Cryogenic liquefaction cools hydrogen to -253 C (20 K), increasing density to 70.8 kg/m3 at an energy penalty of roughly 25% to 30% of its energy content.",
                        "For seasonal terawatt-hour storage near industrial clusters, underground salt caverns store thousands of tonnes of pure hydrogen at 50 to 200 bar with negligible leakage and minimal microbial contamination.",
                    ],
                },
                {
                    "heading": "2. Green Ammonia, Green Steel, and Fuel Cell Applications",
                    "paragraphs": [
                        "Over 80% of global green ammonia projects convert electrolytic green hydrogen and air-separated nitrogen (N2) via the catalytic Haber-Bosch process at 400 C to 450 C and 150 to 250 bar. Green ammonia serves both as zero-carbon nitrogen fertilizer and as an energy-dense liquid hydrogen carrier for intercontinental maritime shipping.",
                        "In primary steelmaking, hydrogen-based Direct Reduced Iron (H2-DRI) replaces coking coal in shaft furnaces to reduce iron ore (Fe2O3) into sponge iron, emitting water vapor instead of CO2 before refining in renewable-powered Electric Arc Furnaces (EAF).",
                        "Proton Exchange Membrane Fuel Cells (PEMFC) recombine hydrogen and oxygen electrochemically to generate electricity at 50% to 60% electrical efficiency with zero tailpipe emissions, while Solid Oxide Fuel Cells (SOFC) achieve over 80% combined heat and power efficiency in stationary microgrids.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "11_grid_scale_battery_and_long_duration_storage.pdf",
            "Grid-Scale BESS Architecture, Grid-Forming Inverters, and Long-Duration Energy Storage",
            [
                {
                    "heading": "1. Battery Energy Storage System (BESS) Architecture and Grid-Forming Inverters",
                    "paragraphs": [
                        "A utility-scale Battery Energy Storage System integrates electrochemical battery racks, a Battery Management System (BMS) monitoring cell voltage and temperature, an HVAC thermal management loop, and bidirectional Power Conversion System (PCS) inverters.",
                        "Safety standards such as UL 9540A and NFPA 855 mandate thermal runaway propagation testing, off-gas ventilation, and fire-suppression barriers between containerized battery enclosures. LFP cells exhibit a thermal runaway onset temperature around 270 C compared with roughly 210 C for high-nickel NMC cells.",
                        "Traditional grid-following inverters rely on an external grid voltage reference to inject current, which weakens grid stability as synchronous coal/gas turbines retire. Grid-forming (GFM) inverters synthesize voltage and frequency internally, providing virtual synchronous inertia, fault-current ride-through, and microgrid islanding.",
                    ],
                },
                {
                    "heading": "2. Emerging Long-Duration Energy Storage (LDES) Technologies",
                    "paragraphs": [
                        "Multi-day weather lulls require Long-Duration Energy Storage (LDES) spanning 10 to 100+ hours where lithium-ion capital costs become prohibitive. Iron-Air batteries reversibly oxidize abundant iron pellets in alkaline electrolyte using ambient oxygen, targeting 100-hour storage at roughly one-tenth the capital cost per kWh of lithium-ion systems.",
                        "Adiabatic Compressed Air Energy Storage (A-CAES) compresses air into underground caverns during surplus generation while capturing compression heat in thermal beds to preheat expanding air through turbines during discharge, raising round-trip efficiency to 65% to 70% without natural gas co-firing.",
                        "Thermal Energy Storage (TES) using electrically heated refractory bricks, molten salts, or high-temperature sand silos stores cheap midday solar power at 600 C to 1,200 C for industrial process steam with over 90% thermal efficiency.",
                    ],
                },
            ],
        )
    )

    created.append(
        _build_pdf(
            "12_global_renewable_policies_and_grid_integration.pdf",
            "National Clean Energy Targets, Transmission Corridors, and Market Integration Mechanisms",
            [
                {
                    "heading": "1. India's 500 GW Roadmap, Green Energy Corridors, and Regulatory Mandates",
                    "paragraphs": [
                        "India has committed to achieving 500 GW of non-fossil-fuel electricity generation capacity and 50% cumulative installed power capacity from non-fossil sources by 2030. To evacuate power from solar-rich Rajasthan and Gujarat and wind-rich Tamil Nadu, the Green Energy Corridor (GEC) builds high-capacity intra-state and inter-state transmission lines.",
                        "The Inter-State Transmission System (ISTS) charge waiver exempts qualifying solar, wind, pumped storage, and green hydrogen renewable projects from wheeling charges across state borders, lowering delivered clean power tariffs for commercial and industrial buyers.",
                        "Renewable Purchase Obligations (RPO) and Renewable Generation Obligations (RGO) require electricity distribution utilities and new thermal generators to procure a rising annual minimum percentage of renewable energy, reinforced by the Approved List of Models and Manufacturers (ALMM) quality mandate for domestic solar modules.",
                    ],
                },
                {
                    "heading": "2. Global Clean Energy Markets, Corporate PPAs, and Virtual Power Plants",
                    "paragraphs": [
                        "Under the United States Inflation Reduction Act (IRA), technology-neutral Clean Electricity Production Tax Credits (Section 45Y) and Investment Tax Credits (Section 48E) reward zero-emission generators while domestic content bonus adders incentivize local steel, battery, and solar supply chains.",
                        "The European Union's Carbon Border Adjustment Mechanism (CBAM) places a carbon price on imported steel, aluminum, cement, fertilizers, and hydrogen, motivating global exporters to sign long-term renewable Corporate Power Purchase Agreements (PPAs).",
                        "Virtual Power Plants (VPPs) aggregate thousands of distributed rooftop solar arrays, home batteries, smart EV chargers, and commercial HVAC loads via cloud software to bid flexible capacity and frequency regulation directly into wholesale electricity markets, cutting renewable curtailment.",
                    ],
                },
            ],
        )
    )

    return created


if __name__ == "__main__":
    paths = generate_all_sample_pdfs()
    for p in paths:
        print(f"Generated: {p}")
