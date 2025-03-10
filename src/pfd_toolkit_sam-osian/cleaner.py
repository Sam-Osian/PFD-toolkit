class Cleaner:
    """
    LLM-powered cleaning agent for Prevention of Future Death Reports.
    """
    def __init__(
        self, 
        llm_model: str = 'gpt-4o-mini',
        ID: bool = False,
        Date: bool = False,
        CoronerName: bool = True,
        Receiver: bool = True,
        InvestigationAndInquest: bool = True,
        CircumstancesofDeath: bool = True,
        MattersOfConcern: bool = True,
        verbose: bool = False
        ) -> None:
        """ 
        Initialises the cleaner.
        
        :param llm_model: The OpenAI LLM model to use for cleaning.
        :param ID: Whether to clean the ID field.
        :param Date: Whether to clean the Date field.
        :param CoronerName: Whether to clean the CoronerName field.
        :param Receiver: Whether to clean the Receiver field.
        :param InvestigationAndInquest: Whether to clean the InvestigationAndInquest field.
        :param CircumstancesofDeath: Whether to clean the CircumstancesofDeath field.
        :param MattersOfConcern: Whether to clean the MattersOfConcern field.
        :param verbose: Whether to print verbose output.
        """
        self.llm_model = llm_model
        self.ID = ID
        self.Date = Date
        self.DeceasedName = DeceasedName
        self.CoronerName = CoronerName
        self.Receiver = Receiver
        self.InvestigationAndInquest = InvestigationAndInquest
        self.CircumstancesofDeath = CircumstancesofDeath
        self.MattersOfConcern = MattersOfConcern
        
        